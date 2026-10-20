"""Read a saved scheduler JSONL trace without contacting a server."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="events-*.jsonl file")
    parser.add_argument("--request", help="Request ID or an unambiguous part of it")
    parser.add_argument("--event", help="Only events containing this text")
    args = parser.parse_args()
    for line in args.trace.read_text().splitlines():
        row = json.loads(line)
        request = row.get("request", {})
        if args.request and args.request not in request.get("id", ""):
            continue
        if args.event and args.event.upper() not in row["event"]:
            continue
        print(
            f"\n#{row['seq']} step={row['step']} {row['event']} "
            f"free_blocks={row['free_blocks']}"
        )
        for queue in ("running", "waiting", "kv_holding_waiting", "deferred_waiting"):
            entries = [f"{r['id']}({r['status']})" for r in row.get(queue, [])]
            print(f"  {queue}: {entries}")
        if request:
            print(
                f"  request: {request['id']} {request['status']} "
                f"prompt={request['prompt']} output={request['output']} "
                f"stored={request['stored']} computed={request['computed']} "
                f"in_flight={request['in_flight']}"
            )
            print(f"  blocks: {request['blocks']}")
        excluded = {
            "seq",
            "step",
            "event",
            "pid",
            "time_ns",
            "monotonic_ns",
            "request",
            "running",
            "waiting",
            "kv_holding_waiting",
            "deferred_waiting",
            "free_blocks",
        }
        details = {k: v for k, v in row.items() if k not in excluded}
        if details:
            print("  details: " + json.dumps(details))


if __name__ == "__main__":
    main()
