"""
Security test runner – all 4 RAG trust pillars.

Pillar 1 – Retrieval Trust Boundaries  (RetrievalTrustBoundary)
Pillar 2 – Indirect Prompt Injection    (IndirectInjectionTester)
Pillar 3 – Context Poisoning            (ContextPoisoner.run_poison_suite)
Pillar 4 – Permission Mistakes          (PermissionAuditor)

Usage:
    # Run all 4 pillars (index must already be built):
    python scripts/test_security.py

    # Rebuild index first, then test:
    python scripts/test_security.py --rebuild

    # Run only specific pillars (1-indexed):
    python scripts/test_security.py --pillars 1 3

    # Skip the safety layer to observe raw model exposure:
    python scripts/test_security.py --no-safety

    # Save results to JSON in results/trustlab/:
    python scripts/test_security.py --save

    # Combine flags:
    python scripts/test_security.py --rebuild --save --pillars 2 4
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# ── project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_global_config
from src.utils.logging import setup_logging
from src.models.rag_model import create_rag_model, load_rag_from_index
from src.safety.input_validator import InputValidator
from src.safety.output_filter import OutputFilter
from src.trustlab import (
    RetrievalTrustBoundary,
    IndirectInjectionTester,
    ContextPoisoner,
    PermissionAuditor,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _load_model(rebuild: bool, config):
    documents_dir = PROJECT_ROOT / "data" / "documents"
    if rebuild:
        print("Building RAG index from documents …")
        return create_rag_model(documents_dir=documents_dir, force_reindex=True)
    print("Loading existing RAG index …")
    try:
        return load_rag_from_index()
    except Exception as exc:
        print(f"  Index not found ({exc}). Building now …")
        return create_rag_model(documents_dir=documents_dir, force_reindex=False)


def _safe_asdict(obj) -> dict:
    """Convert a dataclass to dict, skipping non-serialisable fields."""
    try:
        return asdict(obj)
    except Exception:
        return {"repr": str(obj)}


# ── pillar runners ─────────────────────────────────────────────────────────────

def run_pillar_1(query_engine, safety_on: bool) -> dict:
    _section("Pillar 1 – Retrieval Trust Boundaries")
    tester = RetrievalTrustBoundary()
    results = tester.run_all(query_engine)

    passed = sum(1 for r in results if r.passed)
    for r in results:
        icon = "✓" if r.passed else "✗"
        print(f"  [{icon}] {r.scenario:<25}  retrieved={r.num_retrieved}  verdict={r.verdict}")

    print(f"\n  SUMMARY: {passed}/{len(results)} checks passed")
    return {
        "pillar": "retrieval_trust_boundaries",
        "passed": passed,
        "total": len(results),
        "details": [_safe_asdict(r) for r in results],
    }


def run_pillar_2(query_engine, input_validator, safety_on: bool) -> dict:
    _section("Pillar 2 – Indirect Prompt Injection")
    validator = input_validator if safety_on else None
    tester = IndirectInjectionTester()
    results = tester.run_all(query_engine, input_validator=validator)
    tester.print_report(results)

    failed = sum(1 for r in results if r.injection_succeeded)
    blocked = sum(1 for r in results if r.blocked_by_safety)
    passed = len(results) - failed
    return {
        "pillar": "indirect_prompt_injection",
        "passed": passed,
        "failed": failed,
        "blocked_by_safety": blocked,
        "total": len(results),
        "details": [_safe_asdict(r) for r in results],
    }


def run_pillar_3(query_engine, safety_on: bool) -> dict:
    _section("Pillar 3 – Context Poisoning")
    poisoner = ContextPoisoner()
    results = poisoner.run_poison_suite(query_engine)
    poisoner.print_report(results)

    failed = sum(1 for r in results if r.poison_appeared_in_response)
    passed = len(results) - failed
    return {
        "pillar": "context_poisoning",
        "passed": passed,
        "failed": failed,
        "total": len(results),
        "details": [_safe_asdict(r) for r in results],
    }


def run_pillar_4(query_engine, input_validator, output_filter, safety_on: bool) -> dict:
    _section("Pillar 4 – Permission Mistakes")
    validator = input_validator if safety_on else None
    filtr = output_filter if safety_on else None
    auditor = PermissionAuditor()
    report = auditor.run_all(query_engine, input_validator=validator, output_filter=filtr)
    auditor.print_report(report)

    return {
        "pillar": "permission_mistakes",
        "false_negative_rate": report.false_negative_rate,
        "false_positive_rate": report.false_positive_rate,
        "false_negatives": len(report.false_negatives),
        "false_positives": len(report.false_positives),
        "total": len(report.results),
        "details": [_safe_asdict(r) for r in report.results],
    }


# ── final summary ──────────────────────────────────────────────────────────────

def print_summary(pillar_results: list) -> None:
    _section("OVERALL SECURITY SUMMARY")
    for pr in pillar_results:
        pillar = pr.get("pillar", "unknown")
        if "false_negative_rate" in pr:
            fn = pr["false_negative_rate"]
            fp = pr["false_positive_rate"]
            status = "WARNING" if (fn > 0 or fp > 0) else "PASS"
            print(f"  [{status}] {pillar}")
            print(f"          false-negative rate: {fn:.0%}   false-positive rate: {fp:.0%}")
        else:
            passed = pr.get("passed", 0)
            total = pr.get("total", 0)
            failed = total - passed
            status = "PASS" if failed == 0 else "FAIL"
            print(f"  [{status}] {pillar}  {passed}/{total} passed")
    print()


# ── save results ───────────────────────────────────────────────────────────────

def save_results(pillar_results: list, safety_on: bool) -> None:
    results_dir = PROJECT_ROOT / "results" / "trustlab"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "safety_on" if safety_on else "safety_off"
    out_path = results_dir / f"security_test_{ts}_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": ts,
                "safety_on": safety_on,
                "pillars": pillar_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  Results saved -> {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all 4 RAG security test pillars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the RAG index before testing.")
    parser.add_argument(
        "--pillars", nargs="+", type=int, default=[1, 2, 3, 4],
        metavar="N", help="Which pillars to run (1-4). Default: all.",
    )
    parser.add_argument("--no-safety", action="store_true", help="Disable safety layer (shows raw exposure).")
    parser.add_argument("--save", action="store_true", help="Save results JSON to results/trustlab/.")
    args = parser.parse_args()

    config = get_global_config()
    setup_logging(log_level=config.log_level)

    safety_on = not args.no_safety
    print(f"\nAI Trust Lab – RAG Security Test")
    print(f"  Safety layer : {'ON' if safety_on else 'OFF (raw exposure mode)'}")
    print(f"  Pillars      : {args.pillars}")

    # Load model
    model = _load_model(args.rebuild, config)
    query_engine = model.query_engine
    input_validator = InputValidator() if safety_on else None
    output_filter = OutputFilter() if safety_on else None

    pillar_results = []

    if 1 in args.pillars:
        pillar_results.append(run_pillar_1(query_engine, safety_on))

    if 2 in args.pillars:
        pillar_results.append(run_pillar_2(query_engine, input_validator, safety_on))

    if 3 in args.pillars:
        pillar_results.append(run_pillar_3(query_engine, safety_on))

    if 4 in args.pillars:
        pillar_results.append(run_pillar_4(query_engine, input_validator, output_filter, safety_on))

    print_summary(pillar_results)

    if args.save:
        save_results(pillar_results, safety_on)

    # Exit code: non-zero if any pillar has failures
    total_failures = sum(
        pr.get("failed", 0) + pr.get("false_negatives", 0) + pr.get("false_positives", 0)
        for pr in pillar_results
    )
    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
