#!/usr/bin/env python3
"""Gamescope window for the Frame chat backends: Qwen3-0.6B on the CDSP
(default) or LFM3-1.7B on the CPU (FRAME_CHAT_BACKEND=cpu)."""
import json
import os
from pathlib import Path
import subprocess
import threading

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk


HERE=Path(__file__).resolve().parent
PYTHON=Path(os.environ.get("FRAME_LLM_PYTHON", HERE.parent.parent / ".venv/bin/python"))
PROFILES={
    "npu":{"title":"Frame NPU Chat","app_id":"dev.frame.NPUChat","backend":"chat_backend.py","name":"Qwen",
           "hardware":"Qwen3-0.6B · HMX-ready FP16 weights from Q4_0 · CDSP attention and vector ops",
           "intro":"Ask Qwen a question below. Model arithmetic runs on the Frame NPU; the ARM host manages tokens and memory.\n\n",
           "starting":"Starting NPU backend…","ready":"NPU ready · eye tracking stays paused while the chat is open",
           "after_reset":"New chat · NPU ready","error":"NPU error; see backend log","stopped":"NPU backend stopped; see backend log"},
    "cpu":{"title":"Frame LFM3 Chat","app_id":"dev.frame.LFM3Chat","backend":"chat_backend_cpu.py","name":"LFM3",
           "hardware":"LFM3-1.7B · QAD Q4_0 · llama.cpp on the Frame CPU",
           "intro":"Ask LFM3 a question below. It runs on the Frame's CPU through llama.cpp; eye tracking is unaffected.\n\n",
           "starting":"Starting CPU backend…","ready":"LFM3 ready on the CPU",
           "after_reset":"New chat · LFM3 ready","error":"Backend error; see backend log","stopped":"CPU backend stopped; see backend log"},
}
PROFILE=PROFILES[os.environ.get("FRAME_CHAT_BACKEND","npu")]


class Window(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title=PROFILE["title"])
        self.set_default_size(960,720)
        self.worker=None
        self.busy=False
        self.log=open(HERE / "frame-chat-backend.log", "a", buffering=1)

        outer=Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=10)
        for margin in ("top","bottom","start","end"):
            getattr(outer,f"set_margin_{margin}")(18)
        self.set_child(outer)
        title=Gtk.Label(label=PROFILE["title"])
        title.set_xalign(0)
        title.add_css_class("title-2")
        outer.append(title)
        hardware=Gtk.Label(label=PROFILE["hardware"])
        hardware.set_xalign(0)
        hardware.set_wrap(True)
        outer.append(hardware)
        scroll=Gtk.ScrolledWindow(vexpand=True)
        outer.append(scroll)
        view=Gtk.TextView(editable=False,cursor_visible=False,wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.set_left_margin(12);view.set_right_margin(12)
        view.set_top_margin(12);view.set_bottom_margin(12)
        scroll.set_child(view)
        self.view=view
        self.buffer=view.get_buffer()
        self.append(PROFILE["intro"])

        row=Gtk.Box(spacing=8)
        outer.append(row)
        self.entry=Gtk.Entry(hexpand=True,placeholder_text=f"Ask {PROFILE['name']}…")
        self.entry.connect("activate",self.send)
        row.append(self.entry)
        self.send_button=Gtk.Button(label="Send")
        self.send_button.set_sensitive(False)
        self.send_button.connect("clicked",self.send)
        row.append(self.send_button)
        self.reset_button=Gtk.Button(label="New chat")
        self.reset_button.set_sensitive(False)
        self.reset_button.connect("clicked",self.reset)
        row.append(self.reset_button)
        self.tokens=Gtk.SpinButton.new_with_range(1,512,1)
        self.tokens.set_value(128)
        self.tokens.set_tooltip_text("Maximum response tokens")
        row.append(Gtk.Label(label="Max tokens"))
        row.append(self.tokens)

        self.status=Gtk.Label(label=PROFILE["starting"])
        self.status.set_xalign(0)
        self.status.set_wrap(True)
        outer.append(self.status)
        self.connect("close-request",self.close)
        self.start_backend()

    def append(self, text):
        self.buffer.insert(self.buffer.get_end_iter(),text)
        self.view.scroll_to_iter(self.buffer.get_end_iter(),0,False,0,0)
        return False

    def append_async(self,text):
        GLib.idle_add(self.append,text)

    def set_status(self,text):
        GLib.idle_add(self.status.set_text,text)

    def start_backend(self):
        self.worker=subprocess.Popen([str(PYTHON),str(HERE / PROFILE["backend"])],
                                     cwd=HERE,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                     stderr=self.log,text=True,bufsize=1)
        threading.Thread(target=self.read_backend,daemon=True).start()

    def read_backend(self):
        for line in self.worker.stdout:
            try:
                event=json.loads(line)
            except json.JSONDecodeError:
                self.set_status("Invalid backend response; see log")
                continue
            kind=event.get("type")
            if kind=="ready":
                self.set_status(PROFILE["ready"])
                GLib.idle_add(self.send_button.set_sensitive,True)
                GLib.idle_add(self.reset_button.set_sensitive,True)
            elif kind=="status":
                self.set_status(event["value"])
            elif kind=="text":
                self.append_async(event["value"])
            elif kind=="result":
                self.append_async("\n\n")
                self.set_status(event.get("summary") or f"{event['generated_tokens']} tokens · {event['decode_tokens_per_second']:.2f} tokens/s · "
                                f"{event['prefill_seconds']:.1f}s prompt · {event['npu_calls']} NPU calls · "
                                f"{event.get('accepted_tokens', 0)} drafted tokens accepted · "
                                f"{event['next_context_position']}/{event['context_limit']} context tokens · NPU only")
                GLib.idle_add(self.finish)
            elif kind=="reset":
                GLib.idle_add(self.buffer.set_text, "")
                self.set_status(PROFILE["after_reset"])
                GLib.idle_add(self.finish)
            elif kind in ("error","fatal"):
                self.append_async(f"\nError: {event['message']}\n\n")
                self.set_status(PROFILE["error"])
                GLib.idle_add(self.finish)
        self.set_status(PROFILE["stopped"])
        GLib.idle_add(self.send_button.set_sensitive,False)
        GLib.idle_add(self.reset_button.set_sensitive,False)

    def send(self,*_args):
        prompt=self.entry.get_text().strip()
        if self.busy or not prompt or self.worker is None or self.worker.poll() is not None:
            return
        self.entry.set_text("")
        self.busy=True
        self.send_button.set_sensitive(False)
        self.reset_button.set_sensitive(False)
        self.append(f"You: {prompt}\n\n{PROFILE['name']}: ")
        request={"type":"chat","text":prompt,"max_new_tokens":self.tokens.get_value_as_int()}
        try:
            self.worker.stdin.write(json.dumps(request,ensure_ascii=False)+"\n")
            self.worker.stdin.flush()
        except (BrokenPipeError,OSError) as exc:
            self.append(f"\nBackend error: {exc}\n\n")
            self.finish()

    def finish(self):
        self.busy=False
        self.send_button.set_sensitive(True)
        self.reset_button.set_sensitive(True)
        self.entry.grab_focus()
        return False

    def reset(self,*_args):
        if self.busy or self.worker is None or self.worker.poll() is not None:
            return
        self.busy=True
        self.send_button.set_sensitive(False)
        self.reset_button.set_sensitive(False)
        try:
            self.worker.stdin.write('{"type":"reset"}\n')
            self.worker.stdin.flush()
        except (BrokenPipeError,OSError) as exc:
            self.set_status(f"Backend error: {exc}")
            self.finish()

    def close(self,*_args):
        if self.worker and self.worker.poll() is None:
            self.worker.terminate()
        self.log.close()
        return False


class App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id=PROFILE["app_id"])

    def do_activate(self):
        Window(self).present()


if __name__=="__main__":
    App().run()
