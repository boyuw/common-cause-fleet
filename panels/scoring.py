"""Answer parsing, normalisation and scoring.

Scoring is deliberately strict: a response is correct only if the parsed answer
equals the generator's ground truth after the normalisation below.  Normalisation
removes presentation differences (case, whitespace, currency symbols, date
formatting) but never content differences.
"""
from __future__ import annotations

import json
import re
from datetime import date

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _norm_id(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def _norm_amount(s: str) -> str | None:
    t = re.sub(r"[^0-9.]", "", str(s))
    if not t:
        return None
    try:
        return f"{float(t):.2f}"
    except ValueError:
        return None


def _norm_date(s: str) -> str | None:
    t = str(s).strip()
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", t)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            return None
    return None


def parse_exception(text: str, answer_space: list[str]) -> str | None:
    m = re.search(r"ANSWER\s*[:\-]\s*([A-Za-z_]+)", text)
    if m:
        cand = m.group(1).upper()
        if cand in answer_space:
            return cand
    up = text.upper()
    for cat in answer_space:
        if cat in up:
            return cat
    return None


def parse_decision(text: str) -> str | None:
    m = re.search(r"DECISION\s*[:\-]\s*(PAY|HOLD|ESCALATE)", text.upper())
    if m:
        return m.group(1)
    up = text.upper()
    for lab in ("ESCALATE", "HOLD", "PAY"):
        if re.search(rf"\b{lab}\b", up):
            return lab
    return None


def parse_verdict(text: str) -> str | None:
    up = text.upper()
    m = re.search(r"VERDICT\s*[:\-]\s*(COMPLIANT|VIOLATION)", up)
    if m:
        if m.group(1) == "COMPLIANT":
            return "COMPLIANT"
        c = re.search(r"\bC([1-6])\b", up[m.end():])
        return f"C{c.group(1)}" if c else None
    if "COMPLIANT" in up and "VIOLATION" not in up:
        return "COMPLIANT"
    m2 = re.search(r"VIOLATION[^C]{0,20}C([1-6])", up)
    if m2:
        return f"C{m2.group(1)}"
    return None


def parse_extraction(text: str) -> dict | None:
    for blob in _JSON_RE.findall(text):
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    # fallback: field-by-field regex
    fields = {}
    for f in ("invoice_number", "po_reference", "currency", "total_amount", "due_date"):
        m = re.search(rf'"{f}"\s*:\s*"([^"]*)"', text)
        if m:
            fields[f] = m.group(1)
    return fields or None


def normalise_extraction(obj: dict | None) -> dict:
    if not obj:
        return {}
    low = {str(k).strip().lower(): v for k, v in obj.items()}
    return {
        "invoice_number": _norm_id(low.get("invoice_number", "")),
        "po_reference": _norm_id(low.get("po_reference", "")),
        "currency": _norm_id(low.get("currency", "")),
        "total_amount": _norm_amount(low.get("total_amount", "")) or "",
        "due_date": _norm_date(low.get("due_date", "")) or "",
    }


def score(task_type: str, text: str, gold, answer_space: list[str]) -> dict:
    """Return {parsed, correct}."""
    if task_type == "field_extraction":
        parsed = parse_extraction(text)
        norm = normalise_extraction(parsed)
        gold_norm = {k: (gold[k] if k != "total_amount" else f"{float(gold[k]):.2f}")
                     for k in gold}
        correct = bool(norm) and all(norm.get(k) == gold_norm[k] for k in gold_norm)
        return dict(parsed=norm, correct=correct)
    if task_type == "exception_classification":
        parsed = parse_exception(text, answer_space)
    elif task_type == "three_way_match":
        parsed = parse_decision(text)
    elif task_type == "policy_compliance":
        parsed = parse_verdict(text)
    else:
        raise ValueError(task_type)
    return dict(parsed=parsed, correct=(parsed is not None and parsed == gold))


def parse_checker(text: str, task_type: str, answer_space: list[str]) -> dict:
    """Return {decision: accept|correct|unparseable, answer}."""
    up = text.upper()
    m = re.search(r"CORRECT(?:ED)?\s*[:\-]\s*(.+)", text, re.DOTALL)
    if m is None:
        if re.search(r"\bACCEPT\b", up):
            return dict(decision="accept", answer=None)
        return dict(decision="unparseable", answer=None)
    body = m.group(1)
    if task_type == "field_extraction":
        ans = normalise_extraction(parse_extraction(body))
    elif task_type == "exception_classification":
        ans = parse_exception(body, answer_space)
    elif task_type == "three_way_match":
        ans = parse_decision(body)
    else:
        ans = parse_verdict(body)
    return dict(decision="correct", answer=ans)
