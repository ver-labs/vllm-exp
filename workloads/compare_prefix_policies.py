"""Controlled prefix-reuse workload; server policy is selected at server launch."""

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from urllib.request import Request, urlopen

MODEL = "Qwen/Qwen3-0.6B"
SYSTEM = "You are a helpful assistant. Explain clearly and concisely."
INTRO = (
    "Our town library is reviewing its reservation service. Members search the "
    "catalogue, choose a title, and join a waiting list when every copy is on loan. "
    "Staff contact the next member when a returned copy becomes available. "
    "Reserved books remain at the collection desk for three working days. "
)
DETAILS = (
    "Members can update their contact details through an online account. "
    "The service sends email reminders before a reservation expires. Staff help "
    "members who cannot use the website and record telephone reservations in the "
    "same queue. Each title has its own waiting list ordered by reservation time. "
    "A member can cancel a reservation without losing a place on another list. "
    "The library wants shorter waiting times and fewer uncollected books. "
    "Its budget allows small changes to messages and staff procedures, but no "
    "additional building or large software purchase. Staff will review outcomes "
    "weekly using reservation counts, cancellations, and collection delays. "
)
DOCUMENT = INTRO + DETAILS
PROMPTS = {
    "A": (
        "A coastal research team is studying damage to coral reefs. "
        "They record water temperature, rainfall, salinity, and the number of fish "
        "at several sites. Volunteers photograph the same reef sections every "
        "month. Some sites are close to busy harbours, while others are protected "
        "from boats and fishing. The team has a limited budget and wants reliable "
        "comparisons across locations. Explain how they should organize the "
        "observations and avoid misleading conclusions."
    ),
    "B": (
        "A city transport team is reviewing bus reliability. Drivers report "
        "delays at crowded junctions and passengers report missed connections. "
        "The team has arrival records from stops, traffic counts, and anonymous "
        "passenger surveys. Some routes serve schools, others serve hospitals, "
        "and several share a congested bridge. The city can adjust schedules "
        "and stop locations but cannot purchase new buses this year. Explain "
        "how the team should choose improvements and measure their effects."
    ),
    "C": DOCUMENT + "\nSuggest three improvements to the reservation process.",
    "D": DOCUMENT + "\nSuggest three ways to reduce uncollected reservations.",
    "E": INTRO
    + (
        "A separate branch instead uses paper cards and telephone calls. "
        "It has two volunteers each afternoon and no online member accounts. "
        "Suggest three practical improvements for this branch."
    ),
}


def build_prompts(count):
    prompts = dict(PROMPTS)
    if count == 5:
        return prompts
    unrelated = {
        "F": "A beekeeper is tracking how bees pollinate flowering plants.",
        "I": "Astronomers are observing gas clouds where new stars are forming.",
        "L": "An apartment building wants to turn kitchen scraps into compost.",
        "O": "Geologists are studying earthquakes along an active fault line.",
        "R": "A school science club is comparing rechargeable battery designs.",
        "U": "A village is choosing water filters for its community buildings.",
        "X": "Traffic planners are investigating congestion at a busy crossroads.",
    }
    long_questions = {
        "G": "Suggest three improvements to reservation reminder emails.",
        "J": "Explain how staff should handle cancelled reservations.",
        "M": "Suggest three useful weekly measurements for the library.",
        "P": "Explain how the library can support members without internet access.",
        "S": "Suggest three ways to make collection deadlines clearer.",
        "V": "Explain how staff can investigate repeated collection delays.",
        "Y": "Suggest a simple plan to evaluate changes to the service.",
    }
    medium_questions = {
        "H": "Explain how volunteers could organize the paper waiting list.",
        "K": "Suggest three ways to improve telephone reminders.",
        "N": "Explain how volunteers should record cancellations.",
        "Q": "Suggest three ways to make collection times convenient.",
        "T": "Explain how staff can prevent duplicate paper reservations.",
        "W": "Suggest simple measurements for the branch to review weekly.",
    }
    for label in "FGHIJKLMNOPQRSTUVWXY":
        if label in unrelated:
            prompts[label] = unrelated[label] + (
                " The team has limited funding and wants a practical plan. "
                "It can collect observations weekly but cannot hire more staff. "
                "Explain three useful steps, how to compare the results, and "
                "one common mistake to avoid when interpreting the observations."
            )
        elif label in long_questions:
            prompts[label] = DOCUMENT + "\n" + long_questions[label]
        else:
            prompts[label] = (
                INTRO
                + (
                    "A separate branch uses paper cards and telephone calls. "
                    "It has two volunteers each afternoon and no online accounts. "
                )
                + medium_questions[label]
            )
    return prompts


BLOCKERS = {
    f"H{i}": (
        f"For workshop number {i}, write a detailed numbered list of forty "
        "different household objects. For each object explain its purpose and "
        "how it is made. Continue the list without an introduction."
    )
    for i in range(1, 4)
}
MAX_TOKENS = 64
BLOCKER_TOKENS = 128
PRINT_LOCK = Lock()


def send(url, run_id, label, prompt, limit, origin, first_text=None):
    trace_id = f"{run_id}-{label}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": limit,
        "temperature": 0,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-Request-Id": trace_id},
    )
    start = time.perf_counter()
    first = None
    answer = ""
    usage = {}
    finish = None
    error = None
    done = False
    try:
        with urlopen(request, timeout=180) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content") or ""
                    if text and first is None:
                        first = time.perf_counter()
                        if first_text is not None:
                            first_text.set()
                    answer += text
                    finish = choice.get("finish_reason") or finish
        if not done or finish is None:
            raise RuntimeError("Incomplete streaming response")
        if usage.get("completion_tokens") != limit:
            raise RuntimeError(f"Expected {limit} output tokens; got {usage}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    result = {
        "request_id": label,
        "trace_request_id": trace_id,
        "prompt": prompt,
        "answer": answer,
        "max_output_tokens": limit,
        "started_ms": round((start - origin) * 1000, 2),
        "finished_ms": round((end - origin) * 1000, 2),
        "first_text_ms": round((first - start) * 1000, 2) if first else None,
        "total_ms": round((end - start) * 1000, 2),
        "usage": usage,
        "finish_reason": finish,
        "error": error,
    }
    with PRINT_LOCK:
        print(f"{label}: {error or str(result['total_ms']) + ' ms'}", flush=True)
    return result


def summarize(rows, duration_ms):
    good = [row for row in rows if row["error"] is None]
    out = sum(row["usage"].get("completion_tokens", 0) for row in good)
    result = {
        "successful_requests": len(good),
        "failed_requests": len(rows) - len(good),
        "successful_output_tokens": out,
        "duration_ms": round(duration_ms, 2),
        "output_tokens_per_second": round(out / (duration_ms / 1000), 2),
        "requests_per_second": round(len(good) / (duration_ms / 1000), 3),
    }
    for field in ("first_text_ms", "total_ms"):
        values = sorted(row[field] for row in good if row[field] is not None)
        if values:
            result[field] = {
                "median": round(statistics.median(values), 2),
                "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1],
            }
    return result


def inspect_trace(path, run_id, prompts=None):
    prompts = PROMPTS if prompts is None else prompts
    report = {
        "server_policy": None,
        "max_candidates_waiting": 0,
        "candidate_reorders": [],
        "candidate_cache_scores": {},
        "candidate_admission_order": [],
        "protected_fcfs_head": [],
        "waiting_time_safeguard_count": 0,
    }
    candidate_ids = {f"{run_id}-{label}" for label in prompts}

    def is_candidate(request_id):
        return any(
            request_id == key or key + "-" in request_id for key in candidate_ids
        )

    with Path(path).open() as source:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live trace may end with a partially written line.
            if event["event"] == "TRACE_READY":
                report["server_policy"] = event.get("waiting_policy")
            waiting = event.get("waiting", [])
            count = sum(is_candidate(r["id"]) for r in waiting)
            report["max_candidates_waiting"] = max(
                report["max_candidates_waiting"], count
            )
            rid = event.get("request", {}).get("id", "")
            if not is_candidate(rid):
                continue
            label = rid.split(run_id + "-", 1)[-1].split("-", 1)[0]
            if event["event"] == "ADMIT_OR_RESUME":
                report["candidate_admission_order"].append(label)
            if event["event"] == "PREFIX_SELECTED":
                if event["reason"] == "protected_fcfs_head":
                    report["protected_fcfs_head"].append(label)
                if event["reason"] == "waiting_time_safeguard":
                    report["waiting_time_safeguard_count"] += 1
            if event["event"] == "PREFIX_CANDIDATE":
                scores = report["candidate_cache_scores"].setdefault(rid, [])
                if event["cached_tokens"] not in scores:
                    scores.append(event["cached_tokens"])
            if event["event"] == "PREFIX_SELECTED" and event["reordered"]:
                report["candidate_reorders"].append(
                    {
                        "selected": rid,
                        "previous_head": event["previous_head"],
                        "cached_tokens": event["cached_tokens"],
                        "reason": event["reason"],
                        "step": event["step"],
                    }
                )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("fcfs", "prefix_first", "fcfs_head_prefix"),
        required=True,
        help="Result label; select the real policy at server startup",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    trace_options = parser.add_mutually_exclusive_group()
    trace_options.add_argument(
        "--trace", type=Path, help="Optional current events JSONL file"
    )
    trace_options.add_argument(
        "--trace-latest",
        action="store_true",
        help="Inspect the newest trace for this policy",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Print the workload without contacting a server",
    )
    parser.add_argument("--candidates", type=int, choices=(5, 25), default=5)
    args = parser.parse_args()
    prompts = build_prompts(args.candidates)
    concurrency = len(prompts) + len(BLOCKERS)
    manifest = {
        "model": MODEL,
        "system": SYSTEM,
        "warmup": DOCUMENT + "\nSummarize this.",
        "blockers": BLOCKERS,
        "candidates": prompts,
        "candidate_order": list(prompts),
        "candidate_interval_ms": 20,
        "candidate_output_tokens": MAX_TOKENS,
        "blocker_output_tokens": BLOCKER_TOKENS,
        "ignore_eos": True,
        "maximum_client_concurrency": concurrency,
    }
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    run_id = f"prefix-study-{time.time_ns()}"
    folder = Path("results/prefix-comparison") / args.policy
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"controlled-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    if args.trace_latest:
        paths = list(folder.glob("server-trace-*/events-*.jsonl"))
        if not paths:
            raise SystemExit("No trace found. Start this policy in trace mode first.")
        args.trace = max(paths, key=lambda path: path.stat().st_mtime_ns)
    origin = time.perf_counter()
    warmup = send(args.url, run_id, "WARMUP", manifest["warmup"], 1, origin)
    if warmup["error"]:
        raise SystemExit(f"Warm-up failed: {warmup['error']}")
    origin = time.perf_counter()
    rows = []
    overlap = False
    setup_error = None
    candidate_release_ms = None
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        signals = [Event() for _ in BLOCKERS]
        holders = [
            executor.submit(
                send, args.url, run_id, label, prompt, BLOCKER_TOKENS, origin, signal
            )
            for (label, prompt), signal in zip(BLOCKERS.items(), signals)
        ]
        deadline = time.perf_counter() + 60
        ready = all(
            signal.wait(max(0, deadline - time.perf_counter())) for signal in signals
        )
        jobs = list(holders)
        if ready and not any(job.done() for job in holders):
            release = time.perf_counter()
            candidate_release_ms = round((release - origin) * 1000, 2)
            for index, (label, prompt) in enumerate(prompts.items()):
                delay = release + index * 0.020 - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                jobs.append(
                    executor.submit(
                        send, args.url, run_id, label, prompt, MAX_TOKENS, origin
                    )
                )
            overlap = not any(job.done() for job in holders)
        else:
            setup_error = "Blockers did not overlap long enough; candidates not sent"
        for job in as_completed(jobs):
            rows.append(job.result())
    duration_ms = (time.perf_counter() - origin) * 1000
    candidates = [row for row in rows if row["request_id"] in prompts]
    # Include waiting from the common release, rather than measuring only decode.
    candidate_span = (
        (max(row["finished_ms"] for row in candidates) - candidate_release_ms)
        if candidates
        else None
    )
    trace = inspect_trace(args.trace, run_id, prompts) if args.trace else None
    policy_matches = trace is None or trace["server_policy"] == args.policy
    queue_confirmed = trace is None or trace["max_candidates_waiting"] == len(prompts)
    result = {
        "workload": "controlled-prefix-v2-ay"
        if len(prompts) == 25
        else "controlled-prefix-v1",
        "run_id": run_id,
        "policy_label": args.policy,
        "policy_verified_from_trace": trace is not None and policy_matches,
        "url": args.url,
        "manifest": manifest,
        "warmup": warmup,
        "blockers_unfinished_after_all_submissions": overlap,
        "setup_error": setup_error,
        "candidate_release_ms": candidate_release_ms,
        "valid_run": overlap
        and policy_matches
        and queue_confirmed
        and len(candidates) == len(prompts)
        and all(row["error"] is None for row in rows),
        "batch_summary": summarize(rows, duration_ms),
        "candidate_summary": summarize(candidates, candidate_span)
        if candidates
        else None,
        "diagnostic_selection_demonstrated": (
            bool(trace["candidate_reorders"])
            if trace and args.policy == "prefix_first"
            else None
        ),
        "trace_analysis": trace,
        "requests": sorted(rows, key=lambda r: r["request_id"]),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "valid_run": result["valid_run"],
                "candidate_summary": result["candidate_summary"],
                "trace_analysis": trace,
            },
            indent=2,
        )
    )
    print(f"Saved: {output.resolve()}")
    if not overlap:
        print("Setup did not hold all running slots through candidate submission.")
    if trace and trace["max_candidates_waiting"] < len(prompts):
        print(f"Trace did not confirm all {len(prompts)} candidates waiting together.")
    if trace and args.policy == "prefix_first" and not trace["candidate_reorders"]:
        print("No candidate reordered: inspect scores and waiting-time safeguard.")
    if not result["valid_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
