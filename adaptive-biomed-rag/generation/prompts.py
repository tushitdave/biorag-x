"""Prompt builders for grounded biomedical answer generation.

The prompt is deliberately strict: the model must answer ONLY from the numbered
evidence passages and attach passage-id citations to every claim. This keeps the
output faithful and machine-checkable by the citation validator. The passages are
labelled with their real ``passage_id`` (parent id) so citations map straight back
to retrieved evidence.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful biomedical question-answering assistant. "
    "You answer ONLY using the supplied evidence passages and never use outside "
    "knowledge. Every factual claim must cite the passage id(s) that support it. "
    "If the evidence is insufficient, say so explicitly. "
    "Respond with a single valid JSON object and nothing else."
)

# The JSON schema the model must emit. Kept small + explicit for reliable parsing.
_SCHEMA_HINT = (
    '{\n'
    '  "answer": "<concise answer grounded only in the evidence>",\n'
    '  "claims": [\n'
    '    {"text": "<one factual statement>", "citations": ["<passage_id>", ...]}\n'
    '  ],\n'
    '  "evidence_status": "sufficient | weak | insufficient"\n'
    '}'
)


def build_evidence_block(passages: list[dict]) -> str:
    """Render the numbered evidence block using real passage ids as citation keys."""
    lines = []
    for p in passages:
        pid = str(p.get("parent_passage_id") or p.get("passage_id") or p.get("chunk_id"))
        text = (p.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{pid}] {text}")
    return "\n".join(lines)


def build_grounded_prompt(question: str, passages: list[dict]) -> str:
    """Construct the strict grounded-generation user prompt."""
    evidence = build_evidence_block(passages)
    valid_ids = ", ".join(
        sorted({str(p.get("parent_passage_id") or p.get("passage_id") or p.get("chunk_id"))
                for p in passages})
    )
    return (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE PASSAGES (cite by the bracketed passage id):\n{evidence}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer using ONLY the evidence above.\n"
        "- Break the answer into atomic claims; each claim must list the passage "
        "id(s) that directly support it.\n"
        f"- Only cite ids from this set: [{valid_ids}].\n"
        "- If the evidence does not answer the question, set evidence_status to "
        "\"insufficient\" and keep the answer honest about the gap.\n\n"
        f"Return JSON exactly in this shape:\n{_SCHEMA_HINT}"
    )
