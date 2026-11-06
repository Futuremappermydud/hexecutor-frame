#!/usr/bin/env python3
"""Line-oriented NPU chat worker for the Gamescope GTK client."""
import json
import sys

from chat_session import ChatSession


def emit(kind, **fields):
    print(json.dumps({"type": kind, **fields}, ensure_ascii=False), flush=True)


def main():
    try:
        chat = ChatSession()
        emit("ready")
    except Exception as exc:
        emit("fatal", message=str(exc))
        return 1
    # The GUI stops us with SIGTERM, which FrameTransport turns into
    # KeyboardInterrupt; close in `finally` so the DSP session gets its EXIT
    # (AXI clock and eye client restored) instead of being left open.
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line)
                if request.get("type")=="reset":
                    chat.reset()
                    emit("reset")
                    continue
                if request.get("type")!="chat":
                    raise ValueError("unsupported request")
                result=chat.reply(request["text"], max_new_tokens=int(request.get("max_new_tokens", 128)),
                                  on_text=lambda value:emit("text", value=value),
                                  on_status=lambda value:emit("status", value=value))
                emit("result", **result)
            except Exception as exc:
                emit("error", message=str(exc))
                chat.close()
                chat=ChatSession()
    except KeyboardInterrupt:
        pass
    finally:
        chat.close()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
