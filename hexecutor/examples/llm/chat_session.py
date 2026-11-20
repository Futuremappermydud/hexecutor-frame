#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""ChatML token management around the whole-token CDSP Qwen engine."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer

from dsp_qwen import MAX_TOK, MAX_VERIFY, MODEL_DEFAULT, GGUF_PY_DEFAULT, QwenDSP


TOKENIZER_DEFAULT = Path.home() / "frame-chat/models/tokenizer.json"
NONTHINK_PREFIX = "<think>\n\n</think>\n\n"
MAX_NEW_TOKENS = 512


DRAFT_NGRAMS = tuple(int(n) for n in os.environ.get("FRAME_QWEN_DRAFT_NGRAMS", "3,2").split(","))


def lookup_draft(history, limit, sizes=DRAFT_NGRAMS):
    """Prompt-lookup drafting: the tokens that followed the most recent earlier
    occurrence of the history's last n tokens (longest n first)."""
    for n in sizes:
        if len(history) <= n:
            continue
        tail = history[-n:]
        for i in range(len(history) - n - 1, -1, -1):
            if history[i:i + n] == tail:
                return history[i + n:i + n + limit]
    return []


class ChatSession:
    """Chat on the whole-token CDSP engine. The DSP session (and with it the
    KV cache) stays open for the whole conversation; eye tracking stays
    paused until close()."""
    def __init__(self, model=MODEL_DEFAULT, gguf_python=GGUF_PY_DEFAULT,
                 tokenizer_path=TOKENIZER_DEFAULT):
        self.engine = QwenDSP(model, gguf_python=gguf_python)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.open = False
        self.stop_ids = {self.tokenizer.token_to_id("<|im_end|>"),
                         self.tokenizer.token_to_id("<|endoftext|>")}
        self.reset()

    def reset(self):
        self.engine.reset()
        self.started = False
        self.pending_end = None
        self.turns = []
        self.history = []          # token ids in the DSP's KV cache, in order
        # Speculative decoding (greedy-exact): verify drafted tokens in one pass.
        self.speculate = os.environ.get("FRAME_QWEN_SPECULATE", "1") == "1"

    def close(self):
        if self.open:
            self.engine.__exit__(None, None, None)
            self.open = False

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def reply(self, user_text, *, max_new_tokens=128, on_text=None, on_status=None):
        if not user_text.strip():
            raise ValueError("empty prompt")
        if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS}")
        prefix = "" if not self.started else "\n"
        prompt = prefix + f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n" + NONTHINK_PREFIX
        ids = self._encode(prompt)
        if self.pending_end is not None:
            ids.insert(0, self.pending_end)
        if self.engine.position + len(ids) + max_new_tokens > self.engine.ctx:
            raise ValueError(f"context full ({self.engine.ctx} tokens); start a new chat")
        if not self.open:
            self.engine.__enter__()
            self.open = True
        started = time.monotonic()
        calls_before = self.engine.calls
        output_ids = []
        previous_text = ""
        pending = None
        for index in range(0, len(ids), MAX_TOK):
            chunk = ids[index:index + MAX_TOK]
            if on_status:
                on_status(f"Reading prompt {index + len(chunk)}/{len(ids)}")
            pending, _ = self.engine.step(chunk, logits=index + len(chunk) == len(ids))
        self.history += ids
        prefill_seconds = time.monotonic() - started
        decode_started = time.monotonic()
        drafted = accepted = 0

        def emit(token):
            nonlocal previous_text
            output_ids.append(token)
            decoded = self.tokenizer.decode(output_ids, skip_special_tokens=False)
            if on_text and decoded.startswith(previous_text):
                on_text(decoded[len(previous_text):])
            previous_text = decoded
            if on_status:
                on_status(f"Generated {len(output_ids)} token{'s' if len(output_ids) > 1 else ''}")

        # `pending` is the model's next token, not yet in the KV cache.
        stopped = False
        uncached = False           # the last emitted token is not in the KV cache yet
        while len(output_ids) < max_new_tokens:
            if pending in self.stop_ids:
                self.pending_end = pending
                stopped = True
                break
            emit(pending)
            room = max_new_tokens - len(output_ids)
            if not room:
                uncached = True
                break
            draft = (lookup_draft(self.history + [pending], min(MAX_VERIFY - 1, room))
                     if self.speculate else [])
            batch = [pending] + draft
            start = self.engine.position
            greedy = self.engine.verify(batch)
            take = 0
            while take < len(draft) and draft[take] == greedy[take]:
                if draft[take] in self.stop_ids:
                    stopped = True             # cache the stop token, nothing after it
                    take += 1
                    break
                take += 1
            drafted += len(draft)
            accepted += take
            # Rejected drafts stay in the KV cache past `position` and are
            # overwritten by later tokens.
            self.engine.position = start + 1 + take
            self.history += batch[:1 + take]
            pending = greedy[take - 1] if stopped else greedy[take]
            for token in draft[:take - 1 if stopped else take]:
                emit(token)
            if stopped:
                self.pending_end = None
                break
        if not stopped:
            # Out of tokens: cache the last token if needed, close the turn.
            tail = ([output_ids[-1]] if uncached else []) + [self.tokenizer.token_to_id("<|im_end|>")]
            self.engine.step(tail, logits=False)
            self.history += tail
            self.pending_end = None
        decode_seconds = time.monotonic() - decode_started
        self.started = True
        generated = len(output_ids)
        report = {"prompt_tokens": len(ids), "generated_tokens": generated,
                  "seconds": time.monotonic() - started,
                  "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
                  "decode_tokens_per_second": generated / decode_seconds if decode_seconds else 0.0,
                  "npu_calls": self.engine.calls - calls_before,
                  "drafted_tokens": drafted, "accepted_tokens": accepted,
                  "next_context_position": self.engine.position,
                  "context_limit": self.engine.ctx,
                  "text": previous_text}
        self.turns.append({"user": user_text, "assistant": previous_text, "report": report})
        return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,default=MODEL_DEFAULT)
    p.add_argument("--gguf-python",type=Path,default=GGUF_PY_DEFAULT)
    p.add_argument("--tokenizer",type=Path,default=TOKENIZER_DEFAULT)
    p.add_argument("--prompt",required=True)
    p.add_argument("--max-new-tokens",type=int,default=64)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args()
    chat=ChatSession(args.model,args.gguf_python,args.tokenizer)
    result=chat.reply(args.prompt,max_new_tokens=args.max_new_tokens,
                      on_text=lambda value:print(value,end="",flush=True),
                      on_status=lambda value:print(f"\n[{value}]",flush=True))
    chat.close()
    print("\n"+json.dumps(result),flush=True)
    args.report.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
