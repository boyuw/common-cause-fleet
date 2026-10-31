"""Panel runner.

Design (fixed before execution, see README):

  slot grid   every case is run at 5 slots.  A slot fixes a (seed, paraphrase)
              pair, so slot identity is comparable across configurations.
              M1 occupies all five slots; M3 occupies all five; M2 occupies
              slots 1 and 3 (its two slowest-cost positions, sufficient for a
              matched same-family contrast and for the mixed panels).

  panels      nested, derived from the grid, identical slot positions:
                same_model   k=5 -> M1 at 0..4
                same_family  k=5 -> M1,M2,M1,M2,M1  (slots 0..4)
                cross_family k=5 -> M1,M3,M1,M3,M1
              k=3 uses slots 0..2, k=1 uses slot 0, so the k=1 baseline is
              identical across configurations by construction.

  checkers    author = slot 0 of M1 or M3; four author-checker cells.

Everything is appended to JSONL as it completes, so the run is resumable.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import ollama_backend as ob
from . import scoring

ROOT = Path(__file__).resolve().parent.parent

M1 = "qwen2.5:14b-instruct"   # qwen2 family, 14.8B
M2 = "qwen2.5-coder:32b"      # qwen2 family, 32.8B  (same family, different size)
M3 = "llama3.1:8b"            # llama family, 8.0B   (cross-family)

SLOTS = [0, 1, 2, 3, 4]
# which slots each model occupies in the measurement grid
GRID = {M1: [0, 1, 2, 3, 4], M3: [0, 1, 2, 3, 4], M2: [1, 3]}

PANELS = {
    "same_model": [M1, M1, M1, M1, M1],
    "same_family": [M1, M2, M1, M2, M1],
    "cross_family": [M1, M3, M1, M3, M1],
}

# author-checker cells: (name, author model, author slot, checker model)
CHECKER_CELLS = [
    ("same-same", M1, 0, M1),
    ("same-cross", M1, 0, M3),
    ("cross-same", M3, 0, M1),
    ("cross-cross", M3, 0, M3),
]

CHECKER_TEMPLATE = (
    "You are reviewing a colleague's work.\n\n"
    "TASK GIVEN TO YOUR COLLEAGUE\n{instruction}\n\n"
    "RECORD\n{body}\n\n"
    "YOUR COLLEAGUE'S ANSWER\n{answer}\n\n"
    "Check that answer against the record and the rules. If it is correct, accept it. "
    "If it is wrong, correct it.\n"
    "Reply with exactly one line, either\nACCEPT\nor\n"
    "CORRECT: <the corrected answer, in the same format your colleague was asked to use>"
)


def slot_seed(slot: int) -> int:
    return 1103 + 37 * slot


def load_battery(path: Path | None = None) -> list[dict]:
    path = path or (ROOT / "tasks" / "battery.jsonl")
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def _done_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(r["key"])
    return keys


def run_grid(cases: list[dict], out_path: Path, max_workers: int = 1,
             limit: int | None = None) -> None:
    """Panel members: batched by model so Ollama never reloads between calls."""
    done = _done_keys(out_path)
    if limit:
        cases = cases[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_done = 0
    total = sum(len(GRID[m]) for m in GRID) * len(cases)
    with out_path.open("a") as fh:
        for model, slots in GRID.items():
            for other in GRID:
                if other != model:
                    ob.unload(other)
            for case in cases:
                for slot in slots:
                    key = f"grid|{case['case_id']}|{model}|{slot}"
                    if key in done:
                        continue
                    prompt = case["prompt_paraphrases"][slot]
                    res = ob.generate(model, prompt, seed=slot_seed(slot),
                                      temperature=0.8, num_predict=72)
                    sc = scoring.score(case["task_type"], res["text"], case["gold"],
                                       case["answer_space"])
                    rec = dict(
                        key=key, kind="grid", case_id=case["case_id"],
                        task_type=case["task_type"], difficulty=case["difficulty"],
                        model=model, slot=slot, seed=slot_seed(slot),
                        paraphrase=slot, prompt_sha=None,
                        raw=res["text"], parsed=sc["parsed"], gold=case["gold"],
                        correct=bool(sc["correct"]), latency_s=res["wall_s"],
                        eval_count=res["eval_count"], error=res["error"],
                        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_done += 1
                    if n_done % 20 == 0:
                        el = time.time() - t_start
                        rate = n_done / el
                        print(f"  [{model}] {n_done}/{total} done, "
                              f"{rate*60:.1f}/min, eta {(total-n_done)/max(rate,1e-9)/60:.0f} min",
                              flush=True)


def run_checkers(cases: list[dict], grid_path: Path, out_path: Path,
                 limit: int | None = None) -> None:
    """Author-checker cells, batched by checker model."""
    if limit:
        cases = cases[:limit]
    # author answers from the grid
    authors: dict[tuple[str, int], dict] = {}
    with grid_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            authors[(r["case_id"], r["model"], r["slot"])] = r
    done = _done_keys(out_path)
    by_case = {c["case_id"]: c for c in cases}

    tasks = []
    for cname, amodel, aslot, cmodel in CHECKER_CELLS:
        for case in cases:
            key = f"checker|{case['case_id']}|{cname}"
            if key in done:
                continue
            a = authors.get((case["case_id"], amodel, aslot))
            if a is None or a.get("error"):
                continue
            tasks.append((key, cname, amodel, aslot, cmodel, case, a))

    tasks.sort(key=lambda t: (t[4], t[1]))  # batch by checker model
    t_start = time.time()
    current = None
    with out_path.open("a") as fh:
        for i, (key, cname, amodel, aslot, cmodel, case, a) in enumerate(tasks, 1):
            if cmodel != current:
                for other in {M1, M2, M3} - {cmodel}:
                    ob.unload(other)
                current = cmodel
            prompt = CHECKER_TEMPLATE.format(
                instruction=case["instruction"], body=case["body"], answer=a["raw"])
            res = ob.generate(cmodel, prompt, seed=9001 + aslot, temperature=0.2,
                              num_predict=96)
            pc = scoring.parse_checker(res["text"], case["task_type"], case["answer_space"])
            ans = pc["answer"]
            checker_correct = False
            if pc["decision"] == "correct" and ans is not None:
                if case["task_type"] == "field_extraction":
                    checker_correct = all(ans.get(k) == case["gold"][k] for k in case["gold"])
                else:
                    checker_correct = (ans == case["gold"])
            rec = dict(
                key=key, kind="checker", case_id=case["case_id"],
                task_type=case["task_type"], difficulty=case["difficulty"],
                condition=cname, author_model=amodel, author_slot=aslot,
                checker_model=cmodel, author_correct=a["correct"],
                author_answer=a["raw"], author_parsed=a["parsed"],
                raw=res["text"], decision=pc["decision"], checker_answer=ans,
                checker_correct=checker_correct, gold=case["gold"],
                latency_s=res["wall_s"], error=res["error"],
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 20 == 0:
                el = time.time() - t_start
                print(f"  [checker] {i}/{len(tasks)}, {i/el*60:.1f}/min, "
                      f"eta {(len(tasks)-i)/(i/el)/60:.0f} min", flush=True)


def write_manifest(path: Path) -> None:
    fam = ob.model_family()
    payload = dict(
        models={M1: fam.get(M1), M2: fam.get(M2), M3: fam.get(M3)},
        grid={k: v for k, v in GRID.items()},
        panels=PANELS,
        checker_cells=CHECKER_CELLS,
        slot_seed={s: slot_seed(s) for s in SLOTS},
        temperature_panel=0.8, temperature_checker=0.2,
        num_predict_panel=72, num_predict_checker=96,
        host=ob.HOST,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
