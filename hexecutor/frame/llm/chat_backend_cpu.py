#!/usr/bin/env python3
"""Line-oriented chat worker for the Gamescope GTK client: LFM3-1.7B on the
Frame CPU through llama.cpp's llama-server (same protocol as chat_backend.py).

The server keeps the conversation's KV cache between turns, applies the
model's own chat template (--jinja) and streams tokens. It listens on
127.0.0.1 only and dies with this process. Knobs:
  FRAME_LFM3_THINKING=1   let the model think before answering (default off)
  FRAME_LFM3_THREADS=4    CPU threads; more threads compete with SteamVR
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOME = Path.home()
SERVER = Path(os.environ.get("FRAME_LFM3_SERVER", HOME / "frame-chat/vendor/llama.cpp-lfm3/build/bin/llama-server"))
MODEL = Path(os.environ.get("FRAME_LFM3_MODEL", HOME / "frame-chat/models/LFM3-1.7B/LFM3-1.7B-QAD-Q4_0.gguf"))
PORT = int(os.environ.get("FRAME_LFM3_PORT", "8765"))
THREADS = int(os.environ.get("FRAME_LFM3_THREADS", "4"))
CONTEXT = int(os.environ.get("FRAME_LFM3_CONTEXT", "8192"))
THINKING = os.environ.get("FRAME_LFM3_THINKING") == "1"
URL = f"http://127.0.0.1:{PORT}"
# generation_config.json of LiquidAI/LFM3-1.7B
SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1}


def emit(kind, **fields):
    print(json.dumps({"type": kind, **fields}, ensure_ascii=False), flush=True)


def die_with_parent():
    """Child pre-exec: SIGTERM the server if this backend dies (PR_SET_PDEATHSIG)."""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)


def start_server(log):
    if not SERVER.exists():
        raise RuntimeError(f"llama-server is not built: {SERVER}")
    if not MODEL.exists():
        raise RuntimeError(f"model not found: {MODEL}")
    args = [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
            "-t", str(THREADS), "-c", str(CONTEXT), "-np", "1", "--jinja", "--no-webui",
            "--reasoning-format", "deepseek"]
    proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, preexec_fn=die_with_parent)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with status {proc.returncode}; see the backend log")
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=2) as response:
                if response.status == 200:
                    return proc
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.25)
    proc.terminate()
    raise RuntimeError("llama-server did not become ready within 120 s")


def stream_chat(messages, max_tokens, on_text, on_status):
    body = {"messages": messages, "stream": True, "max_tokens": max_tokens, **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": THINKING}, "cache_prompt": True}
    request = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    reply, thinking, timings, started = [], 0, {}, time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("reasoning_content"):
                    thinking += 1
                    on_status(f"Thinking… ({thinking} tokens)")
                if delta.get("content"):
                    reply.append(delta["content"])
                    on_text(delta["content"])
    return "".join(reply), timings, time.monotonic() - started


def main():
    log = open(Path(__file__).resolve().parent / "frame-chat-cpu-server.log", "a", buffering=1)
    # GUI close sends SIGTERM: turn it into a normal exit so the server is stopped.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    server = None
    try:
        try:
            emit("status", value="Loading LFM3-1.7B on the CPU…")
            server = start_server(log)
            emit("ready")
        except Exception as exc:
            emit("fatal", message=str(exc))
            return 1
        messages = []
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("type") == "reset":
                    messages = []
                    emit("reset")
                    continue
                if request.get("type") != "chat":
                    raise ValueError("unsupported request")
                messages.append({"role": "user", "content": request["text"]})
                text, t, seconds = stream_chat(messages, int(request.get("max_new_tokens", 128)),
                                               lambda value: emit("text", value=value),
                                               lambda value: emit("status", value=value))
                messages.append({"role": "assistant", "content": text})
                generated, context = t.get("predicted_n", 0), t.get("prompt_n", 0) + t.get("cache_n", 0)
                emit("result", generated_tokens=generated,
                     decode_tokens_per_second=t.get("predicted_per_second", 0.0),
                     prefill_seconds=t.get("prompt_ms", 0.0) / 1000.0,
                     summary=f"{generated} tokens · {t.get('predicted_per_second', 0.0):.1f} tokens/s · "
                             f"{t.get('prompt_ms', 0.0) / 1000.0:.1f}s prompt · "
                             f"{context + generated}/{CONTEXT} context tokens · CPU, {THREADS} threads")
            except Exception as exc:
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                emit("error", message=str(exc))
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
