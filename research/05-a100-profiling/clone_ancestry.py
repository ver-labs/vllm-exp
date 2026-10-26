"""Inspect CPU operator ancestry of clones in a trace's first model forward."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("traces", nargs="+", type=Path)
args = parser.parse_args()
result = []
for path in args.traces:
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    scope = min(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation" and e["name"] == "study/model.forward"
        ),
        key=lambda e: e["ts"],
    )
    ops = sorted(
        (
            e
            for e in events
            if e.get("cat") == "cpu_op"
            and (e["pid"], e["tid"]) == (scope["pid"], scope["tid"])
            and scope["ts"] <= e["ts"] < scope["ts"] + scope["dur"]
        ),
        key=lambda e: (e["ts"], -e["dur"]),
    )
    stack, parents = [], Counter()
    for e in ops:
        while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]:
            stack.pop()
        if e["name"] == "aten::clone":
            parents[tuple(item["name"] for item in stack)] += 1
        stack.append(e)
    result.append(
        {
            "trace": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "first_forward_clone_ancestors": [
                {"ancestors": list(k), "clones": v} for k, v in parents.items()
            ],
        }
    )
print(json.dumps(result, indent=2))
