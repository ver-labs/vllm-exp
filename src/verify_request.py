import json
import time
import urllib.request
from pathlib import Path


payload = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "2 + 1?"}],
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
start = time.perf_counter()
with urllib.request.urlopen(request, timeout=120) as response:
    status = response.status
    result = json.load(response)
elapsed = time.perf_counter() - start
record = {
    "request": payload,
    "http_status": status,
    "elapsed_seconds": round(elapsed, 3),
    "response": result,
}

# print(result)
print(json.dumps(result, indent=2))