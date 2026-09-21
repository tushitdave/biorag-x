"""Mermaid diagram builders for the frontend.

Two diagrams:
  * workflow  : the retrieval->answering pipeline as a top-down flowchart with
    clickable stage nodes (the frontend maps node ids to PipelineStage ids).
  * system    : the deployment/component block diagram (frontend + backend +
    artifacts + optional LLM), grouped into subgraphs.

Both return Mermaid source strings (flowchart / block style). Node ids are kept
stable so the UI can bind click handlers to them.
"""
from __future__ import annotations


def build_workflow_mermaid(stage_ids_in_order: list[str] | None = None) -> str:
    """Top-down flowchart of the grounded-RAG workflow."""
    return (
        "flowchart TD\n"
        "  q([User question]) --> ingest[Ingestion & Chunking]\n"
        "  ingest --> represent[Representation: BM25 + Dense MedCPT]\n"
        "  represent --> retrieve{Hybrid Retrieval}\n"
        "  retrieve -->|lexical| bm25[BM25 channel]\n"
        "  retrieve -->|dense| dense[Dense channel]\n"
        "  bm25 --> fuse[RRF Fusion]\n"
        "  dense --> fuse\n"
        "  fuse --> rerank[Reranker]\n"
        "  rerank --> evidence[Evidence Selection MMR]\n"
        "  evidence --> generate[Grounded Generation / Extractive fallback]\n"
        "  generate --> validate[Citation Validation]\n"
        "  validate --> answer([Grounded answer + citations])\n"
        "  classDef data fill:#e3f2fd,stroke:#1976d2;\n"
        "  classDef repr fill:#f3e5f5,stroke:#7b1fa2;\n"
        "  classDef retr fill:#e8f5e9,stroke:#388e3c;\n"
        "  classDef ans fill:#fff3e0,stroke:#f57c00;\n"
        "  class ingest data;\n"
        "  class represent,bm25,dense repr;\n"
        "  class retrieve,fuse,rerank,evidence retr;\n"
        "  class generate,validate ans;\n"
    )


def build_system_mermaid() -> str:
    """Component/block diagram of the running system, grouped by tier."""
    return (
        "flowchart LR\n"
        "  subgraph Client[Frontend - React + TypeScript]\n"
        "    ui[5 Pages: Overview / Pipeline / Chat / Evaluation / System]\n"
        "  end\n"
        "  subgraph API[Backend - FastAPI]\n"
        "    routes[Contract Endpoints]\n"
        "    gen[Generation + Citation Validation]\n"
        "    retr[Hybrid Retrieval]\n"
        "    evalm[Evaluation Metrics]\n"
        "    guard[LLM Guard: switch + budget + cache]\n"
        "  end\n"
        "  subgraph Artifacts[Reused Notebook Artifacts]\n"
        "    passages[(Canonical passages)]\n"
        "    chunks[(Passage units)]\n"
        "    embed[(MedCPT passage vectors)]\n"
        "  end\n"
        "  subgraph Indexes[In-memory Indexes]\n"
        "    bm25idx[[BM25 index]]\n"
        "    denseidx[[FAISS exact index]]\n"
        "  end\n"
        "  llm{{Azure OpenAI - optional, capped}}\n"
        "  ui -->|HTTP JSON| routes\n"
        "  routes --> gen\n"
        "  routes --> retr\n"
        "  routes --> evalm\n"
        "  gen --> guard\n"
        "  retr --> bm25idx\n"
        "  retr --> denseidx\n"
        "  bm25idx --> chunks\n"
        "  denseidx --> embed\n"
        "  denseidx --> chunks\n"
        "  chunks --> passages\n"
        "  guard -.->|only if enabled| llm\n"
        "  guard -.->|else disk cache| Artifacts\n"
        "  classDef client fill:#e3f2fd,stroke:#1976d2;\n"
        "  classDef api fill:#e8f5e9,stroke:#388e3c;\n"
        "  classDef store fill:#fff3e0,stroke:#f57c00;\n"
        "  class ui client;\n"
        "  class routes,gen,retr,evalm,guard api;\n"
        "  class passages,chunks,embed,bm25idx,denseidx store;\n"
    )
