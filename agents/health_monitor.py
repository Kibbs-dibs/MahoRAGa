"""Health-alert probe for MahoRAGa (failure-state monitoring).

Polls the backend readiness endpoint and reports whether the service can serve
queries, which retrieval layers are down, and whether it is running on the
degraded offline LLM fallback. Designed for a scheduler (cron / Windows Task
Scheduler / Kiro Crew cron) so an operator learns a backend or store is down
BEFORE a user does.

Exit codes (so a scheduler can alert on non-zero):
    0  ready and healthy (a real LLM backend reachable)
    1  ready but DEGRADED (serving on offline fallback — no LLM backend reachable)
    2  NOT ready (no retrieval index) or the endpoint returned 503
    3  unreachable (process down / connection refused / timeout)

Usage:
    python -m agents.health_monitor
    python -m agents.health_monitor --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def probe(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0) -> dict:
    """Probe /health/ready and return a normalized status dict.

    Never raises for an unreachable server — that is a reported state, not a
    crash — so a scheduler step always gets a structured result.
    """
    url = base_url.rstrip("/") + "/health/ready"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            # 503 from readiness carries a JSON body describing WHY it is not ready.
            try:
                body = r.json()
            except Exception:
                body = {"status": "unknown", "raw": r.text[:200]}
            body["http_status"] = r.status_code
            return body
    except Exception as exc:
        return {
            "status": "unreachable",
            "error": f"{type(exc).__name__}: {exc}",
            "http_status": None,
        }


def classify(body: dict) -> tuple[int, str]:
    """Map a readiness body to (exit_code, human message)."""
    status = body.get("status")

    if status == "unreachable":
        return 3, f"Backend UNREACHABLE at readiness endpoint ({body.get('error')})."

    if body.get("http_status") == 503 or status == "not_ready":
        return 2, f"Backend NOT READY: {body.get('reason') or 'no retrieval index present.'}"

    if body.get("degraded"):
        configured = body.get("llm_backends_configured") or []
        return 1, (
            "Backend READY but DEGRADED — no LLM backend reachable; answers come "
            f"from the offline fallback and are unverified. Configured: {configured}."
        )

    reachable = body.get("llm_backends_reachable") or []
    return 0, f"Backend healthy — LLM backends reachable: {reachable}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MahoRAGa health-alert probe")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Backend base URL")
    parser.add_argument("--timeout", type=float, default=5.0, help="Probe timeout (s)")
    parser.add_argument("--json", action="store_true", help="Print the raw readiness body as JSON")
    args = parser.parse_args(argv)

    body = probe(args.url, args.timeout)
    code, message = classify(body)

    if args.json:
        print(json.dumps(body, indent=2))
    # Alert line goes to stderr for degraded/down so a scheduler surfaces it.
    stream = sys.stdout if code == 0 else sys.stderr
    print(f"[health_monitor] exit={code} {message}", file=stream)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
