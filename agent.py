import os
from llama_cpp import Llama
import streamlit as st
from tools import (
    open_safari, open_app, open_url, fetch_data,
    fill_pdf, schedule_task
)

class Agent:
    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        seed: int,
        keep_last: int
    ):
        self.history = []
        self.keep_last = keep_last
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            seed=seed,
            n_threads=max(1, os.cpu_count()-1),
            n_batch=256,
            use_metal=True,
            low_vram=True,
            f16_kv=True,
            n_gpu_layers=-1
        )
        # Pre-warm (ignore failures)
        try:
            self.llm.create_completion(prompt="Hello", max_tokens=1)
        except RuntimeError:
            pass

    def analyze_pdf(self, prompt: str, pdf_text: str):
        messages = [
            {"role":"system","content":"You are a PDF assistant."},
            {"role":"user","content":f"PDF:\n{pdf_text}\nQUESTION: {prompt}"}
        ]
        for resp in self.llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.0,
            top_k=1
        ):
            yield resp['choices'][0]['delta'].get('content','')

    def stream_chat(self, prompt: str, pdf_text: str=None):
        lower = prompt.lower()
        if "open safari" in lower:
            open_safari(); yield "✅ Safari opened."; return
        if lower.startswith("open app "):
            name = prompt.split("open app ",1)[1].strip(); open_app(name)
            yield f"✅ Opened {name}."; return
        if lower.startswith("go to ") or lower.startswith("open url "):
            url = prompt.split(None,1)[1]; open_url(url)
            yield f"✅ Navigated to {url}."; return
        if lower.startswith("fetch data from "):
            url = prompt.split("fetch data from ",1)[1].strip(); yield fetch_data(url); return
        if lower.startswith("fill pdf") and pdf_text:
            yield fill_pdf(prompt, st.session_state.get('pdf_path')); return
        if pdf_text and ("pdf" in lower or "summarize" in lower):
            yield from self.analyze_pdf(prompt, pdf_text); return
        if "schedule" in lower or "daily" in lower:
            yield schedule_task(prompt); return
        # Default chat
        truncated = self.history[-self.keep_last*2:]
        messages = [ {"role":e['role'],"content":e['content']} for e in truncated ]
        messages.append({"role":"user","content":prompt})
        for resp in self.llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.0,
            top_k=1
        ):
            yield resp['choices'][0]['delta'].get('content','')