"""Rule-based biomedical entity and relation-cue extraction for the evidence graph.

No trained NER (the research design: rules first, LLM only where uncertain). An entity
is a token that looks biomedical:
  * gene / protein / variant symbols: mixed-case with a digit or 2+ capitals
    (BRCA1, SOX9, TP53, HER2, CDR1as, IL-6, COVID-19)
  * microRNAs: miR-138, miR-21-5p
  * drug / disease words by suffix: dexamethasone, olaparib, trastuzumab, melanoma,
    hepatitis, anemia, fibrosis, cardiomyopathy ...
  * NB04's drug / disease / gene patterns (breast cancer, alzheimer's disease ...)
Keys are normalised (lower case, hyphens removed) so IL-6 / IL6 / il-6 are one node.
Very common generic acronyms are dropped at build time by document frequency.

A relation cue is the first relation trigger in the sentence (NB04 patterns), mapped to
a predicate; sentences without one give CO_OCCURS edges.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from chunking.base import PATTERNS

# symbols with a digit (BRCA1, Sox9, CDR1as) or 2-6 capitals (PARP, IL-6, TNF-alpha); longer
# all-caps words are abstract headings (RESULTS, METHODS), not entities
_SYMBOL = re.compile(r"\b(?:[A-Z][A-Za-z]*\d[A-Za-z0-9]*(?:-\d+[A-Za-z]*)?|[A-Z]{2,6}[a-z]{0,2}\d*"
                     r"(?:-\d+[a-z]?|-alpha|-beta|-gamma)?)\b")
_MIRNA = re.compile(r"\bmi[Rr]-\d+[a-z]?(?:-\d?[35]p)?\b")
_SUFFIX = re.compile(r"\b[a-z]{3,}(?:mab|tinib|nib|ciclib|parib|platin|taxel|rubicin|mycin|cillin|"
                     r"cycline|statin|olol|pril|sartan|azole|parin|vir|sone|olone|gliptin|gliflozin|"
                     r"lukast|tidine|itis|oma|emia|aemia|osis|pathy|plasia|trophy|rrhage|algia)\b")
_MULTI = [PATTERNS[k] for k in ("disease_like", "drug_like", "gene_like", "protein_like")]
_STOP = {"dna", "rna", "mrna", "pcr", "rtpcr", "ci", "or", "hr", "rr", "usa", "uk", "who", "ct",
         "mri", "icu", "iv", "ii", "iii", "iv", "vs", "sd", "se", "et", "al", "fda", "nih", "rct",
         "bmi", "auc", "roc", "ecg", "eeg", "gwas", "snp", "snps", "cdna", "sirna", "shrna",
         "atp", "ph", "id", "ids", "ml", "mg", "kg", "nm", "mm", "min", "max", "aim", "aims",
         "diagnosis", "prognosis", "empathy", "sympathy", "rnas", "dnas",
         # structured-abstract headings in capitals
         "design", "main", "study", "data", "case", "cases", "trial", "review", "level",
         "levels", "type", "group", "groups", "total", "mean", "note", "notes", "key",
         "role", "novel", "new", "does", "cancer", "tumor", "gene", "genes", "protein"} | set(ENGLISH_STOP_WORDS)

# relation trigger (NB04) -> predicate
_PREDICATES = [
    (re.compile(r"\binhibit", re.I), "INHIBITS"),
    (re.compile(r"\bactivat", re.I), "ACTIVATES"),
    (re.compile(r"\btarget", re.I), "TARGETS"),
    (re.compile(r"\bregulat", re.I), "REGULATES"),
    (re.compile(r"\binteracts? with\b", re.I), "INTERACTS_WITH"),
    (re.compile(r"\bexpressed in\b", re.I), "EXPRESSED_IN"),
    (re.compile(r"\b(?:causes?|induces?)\b", re.I), "CAUSES"),
    (re.compile(r"\bprevents?\b", re.I), "PREVENTS"),
    (re.compile(r"\btreat", re.I), "TREATS"),
    (re.compile(r"\b(?:increases?|elevat)", re.I), "INCREASES"),
    (re.compile(r"\b(?:decreases?|reduces?)\b", re.I), "DECREASES"),
    (re.compile(r"\bassociated with\b", re.I), "ASSOCIATED_WITH"),
]


def norm(surface: str) -> str:
    return re.sub(r"[\s\-]+", "", surface.lower())


def entities(text: str) -> dict[str, str]:
    """{normalised key: surface form} of biomedical entities in the text."""
    found: dict[str, str] = {}
    for rx in (_MIRNA, _SYMBOL, _SUFFIX, *_MULTI):
        for m in rx.finditer(text or ""):
            surface = m.group(0)
            key = norm(surface)
            if len(key) < 3 or key in _STOP or key.isdigit():
                continue
            found.setdefault(key, surface)
    return found


def predicate(sentence: str) -> str:
    for rx, name in _PREDICATES:
        if rx.search(sentence):
            return name
    return "CO_OCCURS"
