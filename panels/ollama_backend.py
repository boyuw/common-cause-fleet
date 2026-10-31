"""Minimal Ollama backend. Local only: http://localhost:11434, no API keys."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

HOST = "http://localhost:11434"


def available_models(host: str = HOST, timeout: float = 10.0) -> list[str]:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
        data = json.loads(r.read())
    return [m["name"] for m in data.get("models", [])]


def model_family(host: str = HOST) -> dict[str, dict]:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=10.0) as r:
        data = json.loads(r.read())
    out = {}
    for m in data.get("models", []):
        d = m.get("details", {})
        out[m["name"]] = dict(family=d.get("family"), families=d.get("families"),
                              params=d.get("parameter_size"),
                              quant=d.get("quantization_level"))
    return out


def unload(model: str, host: str = HOST) -> None:
    """Evict a model from memory.  The three-model pool does not fit at once on a
    32 GB host (9.0 + 19.9 + 4.9 GB), so the runner evicts between phases."""
    body = json.dumps({"model": model, "keep_alive": 0, "prompt": ""}).encode()
    try:
        req = urllib.request.Request(f"{host}/api/generate", body,
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        pass


def loaded_models(host: str = HOST) -> list[str]:
    try:
        with urllib.request.urlopen(f"{host}/api/ps", timeout=10) as r:
            d = json.loads(r.read())
        return [m["name"] for m in d.get("models", [])]
    except Exception:
        return []


def generate(model: str, prompt: str, seed: int, temperature: float = 0.8,
             num_predict: int = 96, num_ctx: int = 4096, host: str = HOST,
             retries: int = 3, timeout: float = 600.0) -> dict:
    """Single non-streaming completion. Returns text plus timing metadata."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "top_p": 0.95,
            "repeat_penalty": 1.05,
        },
    }).encode()
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{host}/api/generate", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            return dict(
                text=d.get("response", ""),
                wall_s=round(time.time() - t0, 3),
                eval_count=d.get("eval_count"),
                prompt_eval_count=d.get("prompt_eval_count"),
                total_duration_ns=d.get("total_duration"),
                load_duration_ns=d.get("load_duration"),
                error=None,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return dict(text="", wall_s=None, eval_count=None, prompt_eval_count=None,
                total_duration_ns=None, load_duration_ns=None, error=last)
