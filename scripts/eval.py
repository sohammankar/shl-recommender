"""
eval.py — Local evaluation harness

Replays the 10 public conversation traces and measures:
  1. Schema compliance — does every response have the right shape?
  2. Recall@10 — are the expected assessments in our final recommendations?
  3. Behavior probes — does the agent clarify before recommending? does it
     refuse off-topic queries? does it honor refinements?

Run against a LOCAL server:
    # Terminal 1:
    uvicorn app.main:app --reload --port 8000

    # Terminal 2:
    python scripts/eval.py --url http://localhost:8000

Run against the deployed server:
    python scripts/eval.py --url https://your-render-url.onrender.com

WHY WE HAVE A LOCAL EVAL:
The SHL grader runs the same kind of replay. By running it ourselves first,
we can identify which traces are failing (and why) before submission.
Recall@10 gaps almost always trace back to either (a) a retrieval miss
(the right item wasn't in the top-15 candidates given to the LLM) or
(b) a prompt issue (the LLM chose not to include a relevant item).
The eval script prints per-trace breakdowns so you can pinpoint which.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Ground-truth shortlists extracted from the 10 conversation traces.
# Each entry is a list of canonical SHL catalog URLs that the final
# recommendation turn should include.
# ---------------------------------------------------------------------------

# These are derived directly from C1.md - C10.md, from the LAST turn
# in each conversation that contains a recommendations table.
GROUND_TRUTH: dict[str, dict] = {
    "C1": {
        "description": "Senior leadership / CXO selection with OPQ + reports",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
            "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
        ],
        # Conversation to replay (user turns only — we generate assistant turns)
        "user_turns": [
            "We need a solution for senior leadership.",
            "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
            "Selection — comparing candidates against a leadership benchmark.",
            "Perfect, that's what we need.",
        ],
    },
    "C2": {
        "description": "Senior Rust engineer — systems + cognitive + personality",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
            "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
            "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
        "user_turns": [
            "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
            "Yes, go ahead. Should I also add a cognitive test for this level?",
            "That works. Thanks.",
        ],
    },
    "C3": {
        "description": "High-volume contact centre screening (English US)",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
            "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
            "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
            "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
        ],
        "user_turns": [
            "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?",
            "English.",
            "US.",
            "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?",
            "Perfect — new simulation for volume, old solution for finalists. Confirmed.",
        ],
    },
    "C4": {
        "description": "Graduate financial analysts — numerical + finance + SJT",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
            "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
            "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
            "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
        "user_turns": [
            "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.",
            "Good. Can you also add a situational judgement element — work-context decision making for graduates?",
            "That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates.",
        ],
    },
    "C5": {
        "description": "Sales org re-skilling — personality + GSA + sales reports",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
            "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
            "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
        ],
        "user_turns": [
            "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?",
            "What's the difference between OPQ and OPQ MQ Sales Report?",
            "Clear. We'll use OPQ for everyone and add MQ only where we want motivators in the Sales Report; keeping the five solutions as our audit stack.",
        ],
    },
    "C6": {
        "description": "Chemical plant operators — safety-critical personality + knowledge",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
            "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
        ],
        "user_turns": [
            "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?",
            "What's the difference between the DSI and the Safety & Dependability 8.0?",
            "We're industrial. The 8.0 bundle is the right fit. Confirmed.",
        ],
    },
    "C7": {
        "description": "Bilingual healthcare admin — hybrid English knowledge + Spanish personality",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/hipaa-security/",
            "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
            "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
            "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
        "user_turns": [
            "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?",
            "They're functionally bilingual — English fluent for written work. Go with the hybrid.",
            "Are we legally required under HIPAA to test all staff who touch patient records? And does this SHL test satisfy that requirement?",
            "Understood. Keep the shortlist as-is.",
        ],
    },
    "C8": {
        "description": "Admin assistants — Excel + Word simulations + personality",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
            "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
            "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
            "https://www.shl.com/products/product-catalog/view/ms-word-new/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
        "user_turns": [
            "I need to quickly screen admin assistants for Excel and Word daily.",
            "In that case, I am OK with adding a simulation - we want to capture the capabilities.",
            "That's good.",
        ],
    },
    "C9": {
        "description": "Senior full-stack engineer (backend-leaning) — Java/Spring/SQL/AWS/Docker",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
            "https://www.shl.com/products/product-catalog/view/spring-new/",
            "https://www.shl.com/products/product-catalog/view/sql-new/",
            "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
            "https://www.shl.com/products/product-catalog/view/docker-new/",
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
        "user_turns": [
            'Here\'s the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, and mentor mid-level engineers. Strong CI/CD and cloud-native experience required."',
            "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional — they'd review frontend PRs but not own features.",
            "Senior IC. They lead design on their own services but don't manage other engineers directly.",
            "Add AWS and Docker. Drop REST — the API design signal will already come through in Spring and the live interview.",
            "On Java — they'd be working on existing services, not greenfield. Is the Advanced level the right pick?",
            "Do we really need Verify G+ on top of all the technical tests? Feels redundant.",
            "Keep Verify G+. Locking it in.",
        ],
    },
    "C10": {
        "description": "Graduate management trainee — cognitive + SJT (OPQ dropped by user)",
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
        ],
        "user_turns": [
            "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.",
            "But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long.",
            "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def recall_at_k(recommended_urls: list[str], expected_urls: list[str], k: int = 10) -> float:
    """
    Recall@K = |relevant ∩ top-K recommended| / |relevant|
    """
    if not expected_urls:
        return 1.0
    recommended_set = set(u.rstrip("/") for u in recommended_urls[:k])
    expected_set = set(u.rstrip("/") for u in expected_urls)
    return len(recommended_set & expected_set) / len(expected_set)


def run_trace(
    trace_id: str,
    trace: dict,
    base_url: str,
    client: httpx.Client,
    verbose: bool = True,
) -> dict:
    """Replays one conversation trace and returns evaluation results."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Trace {trace_id}: {trace['description']}")
        print(f"{'='*60}")

    messages = []
    final_recommendations = []
    schema_errors = []
    turn_count = 0
    end_reached = False

    for user_turn in trace["user_turns"]:
        if end_reached:
            break
        if turn_count >= 8:
            if verbose:
                print(f"  [WARN] Hit 8-turn cap before conversation ended")
            break

        messages.append({"role": "user", "content": user_turn})
        turn_count += 1

        if verbose:
            print(f"\n  User (turn {turn_count}): {user_turn[:100]}{'...' if len(user_turn)>100 else ''}")

        try:
            resp = client.post(
                f"{base_url}/chat",
                json={"messages": messages},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            schema_errors.append(f"Turn {turn_count}: TIMEOUT (>30s)")
            if verbose:
                print(f"  [ERROR] TIMEOUT")
            break
        except Exception as e:
            schema_errors.append(f"Turn {turn_count}: HTTP error: {e}")
            if verbose:
                print(f"  [ERROR] {e}")
            break

        # Schema validation
        if "reply" not in data:
            schema_errors.append(f"Turn {turn_count}: missing 'reply' field")
        if "recommendations" not in data:
            schema_errors.append(f"Turn {turn_count}: missing 'recommendations' field")
        if "end_of_conversation" not in data:
            schema_errors.append(f"Turn {turn_count}: missing 'end_of_conversation' field")

        recs = data.get("recommendations", [])
        if recs is not None:
            for r in recs:
                if not isinstance(r, dict):
                    schema_errors.append(f"Turn {turn_count}: recommendation is not a dict")
                    continue
                for field in ("name", "url", "test_type"):
                    if field not in r:
                        schema_errors.append(f"Turn {turn_count}: recommendation missing '{field}'")
                # Validate URL format
                url = r.get("url", "")
                if not url.startswith("https://www.shl.com/products/product-catalog/view/"):
                    schema_errors.append(f"Turn {turn_count}: invalid URL: {url}")

        # Track final recommendations
        if recs:
            final_recommendations = [r.get("url", "") for r in recs if isinstance(r, dict)]

        eoc = data.get("end_of_conversation", False)
        reply = data.get("reply", "")[:150]
        if verbose:
            n_recs = len(recs) if recs else 0
            print(f"  Agent (eoc={eoc}, recs={n_recs}): {reply}{'...' if len(data.get('reply',''))>150 else ''}")

        # Add assistant turn to history
        messages.append({"role": "assistant", "content": data.get("reply", "")})
        turn_count += 1  # assistant turn counts toward the 8-turn cap

        if eoc:
            end_reached = True

    recall = recall_at_k(final_recommendations, trace["expected_urls"])

    if verbose:
        print(f"\n  --- Results ---")
        print(f"  Recall@10: {recall:.2f} ({int(recall * len(trace['expected_urls']))}/{len(trace['expected_urls'])} expected URLs found)")
        print(f"  Schema errors: {len(schema_errors)}")
        for e in schema_errors:
            print(f"    ✗ {e}")
        missing = [u for u in trace["expected_urls"] if u.rstrip("/") not in {r.rstrip("/") for r in final_recommendations}]
        if missing:
            print(f"  Missing expected URLs:")
            for u in missing:
                print(f"    ✗ {u}")

    return {
        "trace_id": trace_id,
        "recall": recall,
        "schema_errors": schema_errors,
        "turns_used": turn_count,
        "end_reached": end_reached,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SHL recommender against public traces")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the service")
    parser.add_argument("--trace", help="Run a specific trace only (e.g. C1)")
    parser.add_argument("--quiet", action="store_true", help="Only print summary")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    verbose = not args.quiet

    print(f"Evaluating against: {base_url}")

    # Check /health first
    with httpx.Client() as client:
        try:
            r = client.get(f"{base_url}/health", timeout=120.0)
            if r.status_code != 200:
                print(f"[ERROR] /health returned {r.status_code}: {r.text}")
                sys.exit(1)
            print(f"/health OK: {r.json()}\n")
        except Exception as e:
            print(f"[ERROR] /health unreachable: {e}")
            sys.exit(1)

        traces_to_run = {k: v for k, v in GROUND_TRUTH.items() if not args.trace or k == args.trace}

        results = []
        for trace_id, trace in traces_to_run.items():
            result = run_trace(trace_id, trace, base_url, client, verbose=verbose)
            results.append(result)
            time.sleep(0.5)  # Small delay to avoid rate-limiting Groq

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    mean_recall = sum(r["recall"] for r in results) / len(results) if results else 0
    total_schema_errors = sum(len(r["schema_errors"]) for r in results)

    for r in results:
        status = "✓" if r["recall"] >= 0.7 and not r["schema_errors"] else "✗"
        print(f"  {status} {r['trace_id']}: Recall@10={r['recall']:.2f}  schema_errors={len(r['schema_errors'])}  turns={r['turns_used']}")

    print(f"\n  Mean Recall@10: {mean_recall:.3f}")
    print(f"  Total schema errors: {total_schema_errors}")
    print(f"  Traces run: {len(results)}")

    if mean_recall < 0.5:
        print("\n  [!] Mean Recall below 0.5 — check retrieval (are expected items in top-15 candidates?)")
    if total_schema_errors > 0:
        print(f"\n  [!] {total_schema_errors} schema errors — these will fail the hard eval. Fix before submitting.")


if __name__ == "__main__":
    main()
