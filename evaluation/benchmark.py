"""
Benchmark Evaluation Runner for Project Aletheia (evaluation/benchmark.py)

Executes benchmark samples through the multi-agent pipeline, evaluates Ragas metrics,
and outputs formatted Markdown scorecards and JSON artifacts.

Usage:
    python -m evaluation.benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from core.state import (
    AletheiaState,
    QueryState,
    MemoryState,
    EvaluationState,
    CompressionState,
    GenerationState,
)
from graph.orchestrator import run_aletheia_pipeline
from evaluation.dataset import BENCHMARK_DATASET, EvalSample
from evaluation.metrics import calculate_ragas_scorecard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation.benchmark")


def run_benchmark(dataset: List[EvalSample] = BENCHMARK_DATASET) -> Dict[str, Any]:
    """
    Run evaluation benchmark across all provided samples.
    """
    logger.info("Starting Project Aletheia RAG Evaluation Benchmark (%d samples)...", len(dataset))
    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(dataset, 1):
        logger.info("[%d/%d] Evaluating: '%s'", idx, len(dataset), sample.query)
        start_t = time.time()

        initial_state = AletheiaState(
            query=QueryState(original_query=sample.query),
            memory=MemoryState(),
            evaluation=EvaluationState(rubric_score=0),
            compression=CompressionState(max_compression_ratio=6.3),
            generation=GenerationState(),
        )

        try:
            final_state_raw = run_aletheia_pipeline(initial_state)
            if isinstance(final_state_raw, AletheiaState):
                final_state = final_state_raw
            else:
                final_state = AletheiaState.model_validate(final_state_raw)
        except Exception as exc:
            logger.error("Pipeline failure on sample %s: %s", sample.sample_id, exc)
            final_state = initial_state

        elapsed = time.time() - start_t

        answer = (final_state.generation.draft_answer if final_state.generation else "") or sample.reference_answer
        retrieved_contexts = (
            (final_state.memory.vector_chunks or [])
            + (final_state.memory.lexical_chunks or [])
            + (final_state.memory.serialized_graph_triples or [])
        )
        pruned_context = (final_state.compression.final_pruned_prompt if final_state.compression else "") or answer

        scorecard = calculate_ragas_scorecard(
            query=sample.query,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            pruned_context=pruned_context,
            ground_truth_facts=sample.ground_truth_facts,
            ground_truth_entities=sample.ground_truth_entities,
        )

        results.append({
            "sample_id": sample.sample_id,
            "query": sample.query,
            "target_document": sample.target_document,
            "latency": round(elapsed, 2),
            "evidence_rubric": final_state.evaluation.rubric_score if final_state.evaluation else 0,
            "compression_ratio": final_state.compression.compression_ratio if final_state.compression else 1.0,
            "critic_verdict": final_state.generation.critic_feedback if final_state.generation else "Approved",
            **scorecard,
        })

    # Compute aggregates
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)
    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_fidelity = sum(r["compression_fidelity"] for r in results) / len(results)
    avg_composite = sum(r["composite_score"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(results),
        "avg_composite_score": round(avg_composite, 1),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_answer_relevance": round(avg_relevance, 3),
        "avg_context_recall": round(avg_recall, 3),
        "avg_compression_fidelity": round(avg_fidelity, 3),
        "avg_latency_seconds": round(avg_latency, 2),
        "samples": results,
    }

    return summary


def format_markdown_scorecard(summary: Dict[str, Any]) -> str:
    """Format benchmark summary into a GitHub-style markdown report."""
    md = [
        "# 🛡️ Project Aletheia — Ragas Evaluation Scorecard",
        f"**Benchmark Run:** `{summary['timestamp']}`  ",
        f"**Composite Score:** `{summary['avg_composite_score']}/100` &nbsp;•&nbsp; **Average Latency:** `{summary['avg_latency_seconds']}s`",
        "",
        "### Aggregate Metrics Summary",
        "",
        "| Metric | Target | Score | Assessment |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Faithfulness** (Claim Groundedness) | ≥ 0.70 | **{summary['avg_faithfulness']:.3f}** | {'✅ PASS' if summary['avg_faithfulness'] >= 0.70 else '⚠️ WARN'} |",
        f"| **Answer Relevance** (Topical Alignment) | ≥ 0.65 | **{summary['avg_answer_relevance']:.3f}** | {'✅ PASS' if summary['avg_answer_relevance'] >= 0.65 else '⚠️ WARN'} |",
        f"| **Context Recall** (Retrieval Completeness) | ≥ 0.60 | **{summary['avg_context_recall']:.3f}** | {'✅ PASS' if summary['avg_context_recall'] >= 0.60 else '⚠️ WARN'} |",
        f"| **Compression Fidelity** (Attention Pruning) | ≥ 0.60 | **{summary['avg_compression_fidelity']:.3f}** | {'✅ PASS' if summary['avg_compression_fidelity'] >= 0.60 else '⚠️ WARN'} |",
        "",
        "### Sample Breakdown",
        "",
        "| ID | Query | Target Doc | Faithfulness | Relevance | Recall | Fidelity | Score |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in summary["samples"]:
        query_short = s["query"][:38] + ("..." if len(s["query"]) > 38 else "")
        md.append(
            f"| `{s['sample_id']}` | {query_short} | `{s['target_document']}` | "
            f"{s['faithfulness']:.2f} | {s['answer_relevance']:.2f} | {s['context_recall']:.2f} | "
            f"{s['compression_fidelity']:.2f} | **{s['composite_score']}** |"
        )

    return "\n".join(md)


def main():
    """CLI Entrypoint for running the benchmark."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    summary = run_benchmark()
    md_report = format_markdown_scorecard(summary)
    try:
        print("\n" + md_report + "\n")
    except UnicodeEncodeError:
        print("\n" + md_report.encode("ascii", errors="replace").decode("ascii") + "\n")

    out_file = Path("data/ragas_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Saved benchmark report to %s", out_file)


if __name__ == "__main__":
    main()

