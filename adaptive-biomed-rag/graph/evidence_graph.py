"""Provenance-backed evidence graph and graph-traversal retrieval (research design,
"Graph traversal" route; NB08).

Build (offline, CPU, rules only - no LLM, no vectors):
  nodes     biomedical entities found by graph/extract.py
  mentions  entity -> passages that mention it
  edges     two entities in the same sentence; predicate = the sentence's relation word
            (INHIBITS, TARGETS, ...: a cue, not a verified direction) or CO_OCCURS;
            every edge keeps the passage it came from, so any path can be traced to text
  Entities in more than GENERIC_SHARE of passages are too generic to traverse.

Query (traversal):
  1. seeds = question entities that are in the graph
  2. hop 0: passages mentioning a seed score idf(seed), summed over seeds
     + PAIR_BONUS x mean idf when two seeds share a sentence (x TYPED_WEIGHT if a
       relation verb links them)
  3. hop 1: the MAX_NEIGHBORS strongest neighbours of the seeds (edge support x idf;
     x BRIDGE_BOOST if a neighbour links two seeds) add HOP_DECAY x idf(neighbour) to
     passages mentioning them - this reaches evidence that never names the seed
  Passages are ranked by that score; each keeps the path that got it there.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from chunking.base import sentences
from graph.extract import _STOP, entities, norm, predicate

GENERIC_SHARE = 0.03      # entity in > 3% of passages: too generic to traverse
MAX_PER_SENTENCE = 12     # entity cap per sentence (limits edges from lists)
PAIR_BONUS = 0.5          # two seeds in one sentence
TYPED_WEIGHT = 1.5        # edge with a relation verb vs plain co-occurrence
MAX_NEIGHBORS = 8         # hop-1 entities followed
HOP_DECAY = 0.5           # hop-1 evidence counts half a seed
BRIDGE_BOOST = 2.0        # neighbour connected to two or more seeds
MAX_PATHS = 4             # reasons kept per passage
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(passages: list[tuple[str, str]], out_dir: Path,
          log: Callable[[str], None] = print) -> dict:
    surfaces: dict[str, Counter] = defaultdict(Counter)
    mentions: Counter = Counter()            # (entity, pid) -> sentences
    edges: Counter = Counter()               # (a, b, predicate, pid) -> sentences
    for i, (pid, text) in enumerate(passages):
        for sent in sentences(text):
            found = entities(sent)
            if not found:
                continue
            for k, surf in found.items():
                surfaces[k][surf] += 1
                mentions[(k, pid)] += 1
            keys = sorted(found)[:MAX_PER_SENTENCE]
            if len(keys) > 1:
                pred = predicate(sent)
                for x in range(len(keys)):
                    for y in range(x + 1, len(keys)):
                        edges[(keys[x], keys[y], pred, pid)] += 1
        if i % 5000 == 0:
            log(f"  graph {i:,}/{len(passages):,} passages | {len(surfaces):,} entities")

    m = pd.DataFrame([(e, p, c) for (e, p), c in mentions.items()],
                     columns=["entity", "passage_id", "count"])
    df = m.groupby("entity")["passage_id"].nunique()
    limit = GENERIC_SHARE * len(passages)
    ents = pd.DataFrame({"entity": df.index, "df": df.values})
    ents["surface"] = [surfaces[e].most_common(1)[0][0] for e in ents["entity"]]
    ents["generic"] = ents["df"] > limit
    generic = set(ents.loc[ents["generic"], "entity"])
    e = pd.DataFrame([(a, b, r, p, c) for (a, b, r, p), c in edges.items()
                      if a not in generic and b not in generic],
                     columns=["src", "dst", "predicate", "passage_id", "count"])

    out_dir.mkdir(parents=True, exist_ok=True)
    ents.to_parquet(out_dir / "entities.parquet", index=False)
    m.to_parquet(out_dir / "mentions.parquet", index=False)
    e.to_parquet(out_dir / "edges.parquet", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passages": len(passages), "entities": int(len(ents)),
        "generic_entities": int(len(generic)), "mentions": int(len(m)),
        "edges": int(e[["src", "dst"]].drop_duplicates().shape[0]),
        "edge_provenance_rows": int(len(e)),
        "typed_edges": int((e["predicate"] != "CO_OCCURS").sum()),
        "params": {"generic_share": GENERIC_SHARE, "max_per_sentence": MAX_PER_SENTENCE},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
class EvidenceGraph:
    def __init__(self, graph_dir: Path):
        ents = pd.read_parquet(graph_dir / "entities.parquet")
        self.manifest = json.loads((graph_dir / "manifest.json").read_text())
        n = self.manifest["passages"]
        usable = ents[~ents["generic"]]
        self.idf = {k: math.log(n / d) for k, d in zip(usable["entity"], usable["df"])}
        self.surface = dict(zip(ents["entity"], ents["surface"]))
        m = pd.read_parquet(graph_dir / "mentions.parquet")
        m = m[m["entity"].isin(self.idf)]
        self.mentions = m.groupby("entity")["passage_id"].agg(list).to_dict()
        self.edges = pd.read_parquet(graph_dir / "edges.parquet")

    def seeds(self, question: str) -> list[str]:
        """Question entities in the graph: rule matches, plus plain words / word pairs
        typed in lower case ('brca1', 'ovarian cancer'). A plain word only counts if it
        has a digit or the graph knows it in lower case, so 'Does' never matches a
        heading like 'DOES'."""
        found = list(entities(question))
        words = _WORD.findall(question)
        grams = [norm(w) for w in words] + [norm(f"{a} {b}") for a, b in zip(words, words[1:])]
        for g in grams:
            if g in self.idf and g not in _STOP and len(g) > 2 and \
                    (any(c.isdigit() for c in g) or not self.surface[g].isupper()):
                found.append(g)
        return [k for k in dict.fromkeys(found) if k in self.idf]

    def search(self, question: str, top_k: int | None = None) -> tuple[list[dict], dict]:
        """Passages ranked by traversal score, and a trace of the traversal."""
        seeds = self.seeds(question)
        info = {"seeds": [self.surface[s] for s in seeds], "neighbors": [], "hits": 0}
        if not seeds:
            info["note"] = "no biomedical entity from the question is in the graph"
            return [], info

        score: dict[str, float] = defaultdict(float)
        why: dict[str, list[str]] = defaultdict(list)
        for s in seeds:
            for p in self.mentions.get(s, []):
                score[p] += self.idf[s]
                why[p].append(self.surface[s])

        e = self.edges
        touch = e[e["src"].isin(seeds) | e["dst"].isin(seeds)]
        seedset = set(seeds)
        # two seeds in one sentence of a passage
        both = touch[touch["src"].isin(seedset) & touch["dst"].isin(seedset)]
        for a, b, r, p in both[["src", "dst", "predicate", "passage_id"]].itertuples(index=False):
            w = PAIR_BONUS * (self.idf[a] + self.idf[b]) / 2
            score[p] += w * (TYPED_WEIGHT if r != "CO_OCCURS" else 1.0)
            why[p].append(f"{self.surface[a]} —{r.lower().replace('_', ' ')}— {self.surface[b]}")

        # hop 1: strongest neighbours
        one = touch[~(touch["src"].isin(seedset) & touch["dst"].isin(seedset))].copy()
        one["seed"] = one["src"].where(one["src"].isin(seedset), one["dst"])
        one["nbr"] = one["dst"].where(one["src"].isin(seedset), one["src"])
        one = one[one["nbr"].isin(self.idf)]
        one["w"] = (one["predicate"] != "CO_OCCURS").map({True: TYPED_WEIGHT, False: 1.0})
        if len(one):
            g = one.groupby("nbr").agg(support=("w", "sum"), seeds=("seed", "nunique"))
            g["strength"] = g["support"] * g.index.map(self.idf) * \
                g["seeds"].map(lambda k: BRIDGE_BOOST if k > 1 else 1.0)
            top = g.sort_values("strength", ascending=False).head(MAX_NEIGHBORS)
            for nbr, row in top.iterrows():
                via = one[one["nbr"] == nbr]
                seed = via["seed"].mode().iat[0]
                pred = via["predicate"].mode().iat[0].lower().replace("_", " ")
                path = f"{self.surface[seed]} —{pred}— {self.surface[nbr]}"
                info["neighbors"].append({"entity": self.surface[nbr], "via": self.surface[seed],
                                          "relation": pred, "bridge": bool(row["seeds"] > 1),
                                          "support": int(via["passage_id"].nunique())})
                for p in self.mentions.get(nbr, []):
                    score[p] += HOP_DECAY * self.idf[nbr]
                    why[p].append(path)

        ranked = sorted(score, key=score.get, reverse=True)
        if top_k:
            ranked = ranked[:top_k]
        info["hits"] = len(score)
        out = [{"parent_passage_id": p, "graph_score": round(score[p], 4),
                "graph_path": "; ".join(list(dict.fromkeys(why[p]))[:MAX_PATHS])} for p in ranked]
        return out, info
