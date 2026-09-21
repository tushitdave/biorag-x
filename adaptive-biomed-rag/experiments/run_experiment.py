"""Run a recipe over a small question sample and compute retrieval metrics.

LLM safety: generation is OFF by default here (retrieval-only evaluation makes
zero LLM calls). When ``with_generation=True`` and the LLM master switch is on,
it makes at most ONE call per question, and never more than the sample size, so
the process-wide budget (12) is respected. Default sample size = SETTINGS.benchmark_size.

Returns an EvaluationReport-compatible dict (leaderboard rows + aggregate metrics).
"""
from __future__ import annotations

from common.config import SETTINGS, load_recipe
from ingestion.loader import sample_questions, gold_ids_for_question
from retrieval.hybrid import retrieve
from evaluation.retrieval_metrics import per_query_metrics, aggregate_metrics


def _question_text(row) -> str:
    for col in ("question", "question_text", "body", "text"):
        if col in row and row[col] is not None:
            return str(row[col])
    return ""


def run_recipe(recipe_name: str, n: int | None = None, top_k: int = 10,
               with_generation: bool = False) -> dict:
    """Evaluate a single recipe on the reproducible sample. Retrieval-only by default."""
    n = n or SETTINGS.benchmark_size
    recipe = load_recipe(recipe_name)
    sample = sample_questions(n)

    per_query: list[dict] = []
    llm_calls_made = 0
    for _, row in sample.iterrows():
        q = _question_text(row)
        gold = set(gold_ids_for_question(row))
        result = retrieve(q, recipe, top_k=top_k)
        retrieved_parents = [
            str(p.get("parent_passage_id") or p.get("passage_id"))
            for p in result.get("passages", [])
        ]
        # Also include the reranked pool for recall@10 (evidence set may be small).
        pool = [str(p.get("parent_passage_id")) for p in result.get("reranked", [])]
        ranked = retrieved_parents + [p for p in pool if p not in retrieved_parents]
        per_query.append(per_query_metrics(ranked, gold))

        if with_generation and SETTINGS.llm_enabled:
            # Guarded single call per question via the generator (chat_json).
            from generation.generator import generate_answer
            resp = generate_answer(q, recipe=recipe_name, top_k=5,
                                   retrieval_result=result)
            if resp.llm_source == "live":
                llm_calls_made += 1

    agg = aggregate_metrics(per_query)
    return {
        "recipe": recipe_name,
        "experiment_id": recipe.get("experiment_id", recipe_name),
        "benchmark_size": int(len(sample)),
        "metrics": agg,
        "per_query": per_query,
        "llm_calls_made": llm_calls_made,
    }


def run_leaderboard(recipe_names: list[str], n: int | None = None,
                    top_k: int = 10) -> list[dict]:
    """Retrieval-only leaderboard across recipes (0 LLM calls). Sorted by nDCG@10."""
    results = [run_recipe(r, n=n, top_k=top_k, with_generation=False) for r in recipe_names]
    results.sort(key=lambda r: r["metrics"].get("ndcg@10", 0.0), reverse=True)
    return results
