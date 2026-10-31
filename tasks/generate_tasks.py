"""Synthetic enterprise-process task battery with generator-assigned difficulty.

Every case carries a difficulty label assigned by construction parameters
(tolerance margins, number of injected distractors, number of simultaneously
failing policy conditions, presence of superseding clauses).  Labels are never
derived from model behaviour.

Four task types, all with objective ground truth computed by a rule engine:

  A  exception_classification  order/invoice exception -> fixed category
  B  field_extraction          five fields from a generated invoice document
  C  three_way_match           PO / receipt / invoice -> PAY | HOLD | ESCALATE
  D  policy_compliance         verdict + violated clause against a written policy

Run:  python3 tasks/generate_tasks.py [--n 34] [--out tasks/battery.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK_TYPES = [
    "exception_classification",
    "field_extraction",
    "three_way_match",
    "policy_compliance",
]
STRATA = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# prompt paraphrases: same case, same required output format, different framing.
# Variant 4 additionally reorders the format instruction ahead of the record.
# ---------------------------------------------------------------------------
PARAPHRASES = [
    "You are a senior accounts-payable specialist. Work carefully and answer from the record alone.",
    "Act as a meticulous financial-controls reviewer examining the following record.",
    "You are an experienced shared-services analyst. Read the record below and reach a decision.",
    "Role: finance operations reviewer. Review the case and produce the required answer.",
    "As an audit-ready AP analyst, evaluate the record below strictly on its stated terms.",
]
N_PARAPHRASE = len(PARAPHRASES)


def render(paraphrase_idx: int, instruction: str, body: str, fmt: str) -> str:
    """Assemble a prompt. Variant 4 puts the output format before the record."""
    head = PARAPHRASES[paraphrase_idx % N_PARAPHRASE]
    if paraphrase_idx % N_PARAPHRASE == 4:
        return (
            f"{head}\n\n{instruction}\n\n{fmt}\n\n"
            f"--- RECORD ---\n{body}\n--- END RECORD ---"
        )
    return (
        f"{head}\n\n{instruction}\n\n--- RECORD ---\n{body}\n--- END RECORD ---\n\n{fmt}"
    )


# ---------------------------------------------------------------------------
# A. exception classification
# ---------------------------------------------------------------------------
EXC_CATEGORIES = [
    "NO_EXCEPTION",
    "PRICE_MISMATCH",
    "QTY_MISMATCH",
    "MISSING_PO",
    "DUPLICATE_INVOICE",
    "CURRENCY_MISMATCH",
]
# highest first
EXC_PRECEDENCE = [
    "MISSING_PO",
    "DUPLICATE_INVOICE",
    "CURRENCY_MISMATCH",
    "PRICE_MISMATCH",
    "QTY_MISMATCH",
]
EXC_INSTRUCTION = (
    "Classify the invoice exception. Apply these rules exactly:\n"
    "- MISSING_PO: the invoice carries no purchase-order reference.\n"
    "- DUPLICATE_INVOICE: the invoice number appears in the list of invoices already paid.\n"
    "- CURRENCY_MISMATCH: the invoice currency differs from the purchase-order currency.\n"
    "- PRICE_MISMATCH: |invoice unit price - PO unit price| / PO unit price exceeds 1%.\n"
    "- QTY_MISMATCH: |invoice quantity - received quantity| / received quantity exceeds 2%.\n"
    "- NO_EXCEPTION: none of the above applies.\n"
    "If more than one rule applies, report the highest precedence:\n"
    "MISSING_PO > DUPLICATE_INVOICE > CURRENCY_MISMATCH > PRICE_MISMATCH > QTY_MISMATCH."
)
EXC_FMT = "Answer with exactly one line:\nANSWER: <CATEGORY>"

VENDORS = [
    "Northwind Components BV", "Atlas Fasteners GmbH", "Cedarline Packaging Ltd",
    "Volta Industrial SA", "Kestrel Metals Inc", "Blue Harbour Logistics",
    "Ridgeway Chemicals AB", "Meridian Optics KK", "Tessera Plastics SpA",
]


def _exc_case(rng: random.Random, difficulty: str, idx: int) -> dict:
    po_qty = rng.choice([40, 60, 80, 100, 120, 150, 200, 250])
    po_price = round(rng.uniform(3.0, 240.0), 2)
    recv_qty = po_qty - rng.choice([0, 0, 0, 1, 2])
    inv_qty = recv_qty
    inv_price = po_price
    cur_po = rng.choice(["EUR", "USD", "GBP"])
    cur_inv = cur_po
    po_ref = f"PO-{rng.randint(10000, 99999)}"
    cited_ref = po_ref          # the reference the invoice prints; None => MISSING_PO
    inv_no = f"INV-{rng.randint(200000, 899999)}"
    paid = [f"INV-{rng.randint(100000, 199999)}" for _ in range(2)]

    flags: list[str] = []
    if difficulty == "easy":
        kind = rng.choice(EXC_PRECEDENCE + ["NO_EXCEPTION"])
        if kind == "PRICE_MISMATCH":
            inv_price = round(po_price * (1 + rng.choice([-1, 1]) * rng.uniform(0.08, 0.15)), 2)
        elif kind == "QTY_MISMATCH":
            inv_qty = int(round(recv_qty * (1 + rng.uniform(0.10, 0.20))))
        elif kind == "MISSING_PO":
            cited_ref = None
        elif kind == "DUPLICATE_INVOICE":
            paid.append(inv_no)
        elif kind == "CURRENCY_MISMATCH":
            cur_inv = rng.choice([c for c in ["EUR", "USD", "GBP"] if c != cur_po])
    elif difficulty == "medium":
        # one genuine anomaly just past tolerance, plus a within-tolerance distractor
        kind = rng.choice(["PRICE_MISMATCH", "QTY_MISMATCH", "DUPLICATE_INVOICE",
                           "CURRENCY_MISMATCH", "NO_EXCEPTION"])

        def _near_qty(base: int, target: float, lo: float, hi: float) -> int:
            """Integer quantity whose relative deviation from `base` lands in [lo, hi]."""
            for _ in range(200):
                cand = int(round(base * rng.uniform(target - 0.01, target + 0.01)))
                if cand != base and lo <= abs(cand - base) / base <= hi:
                    return cand
            return base

        if kind == "PRICE_MISMATCH":
            inv_price = round(po_price * (1 + rng.choice([-1, 1]) * rng.uniform(0.012, 0.021)), 2)
            inv_qty = _near_qty(recv_qty, 1.0, 0.0, 0.019)  # within tolerance
        elif kind == "QTY_MISMATCH":
            inv_qty = _near_qty(recv_qty, 1.03, 0.022, 0.045)
            inv_price = round(po_price * rng.uniform(0.995, 1.005), 2)  # within tolerance
        elif kind == "DUPLICATE_INVOICE":
            paid.append(inv_no)
            inv_price = round(po_price * rng.uniform(0.996, 1.004), 2)
        elif kind == "CURRENCY_MISMATCH":
            cur_inv = rng.choice([c for c in ["EUR", "USD", "GBP"] if c != cur_po])
            inv_qty = _near_qty(recv_qty, 1.0, 0.0, 0.019)
        else:  # clean, with two within-tolerance distractors
            inv_qty = _near_qty(recv_qty, 1.0, 0.0, 0.019)
            inv_price = round(po_price * rng.uniform(0.995, 1.005), 2)
    else:  # hard: >= 2 genuine anomalies, precedence decides
        kinds = rng.sample(["PRICE_MISMATCH", "QTY_MISMATCH", "MISSING_PO",
                            "DUPLICATE_INVOICE", "CURRENCY_MISMATCH"],
                           rng.choice([2, 2, 3]))
        for kind in kinds:
            if kind == "PRICE_MISMATCH":
                inv_price = round(po_price * (1 + rng.choice([-1, 1]) * rng.uniform(0.0105, 0.030)), 2)
            elif kind == "QTY_MISMATCH":
                for _ in range(200):
                    cand = int(round(recv_qty * (1 + rng.uniform(0.021, 0.045))))
                    if cand != recv_qty and abs(cand - recv_qty) / recv_qty > 0.021:
                        inv_qty = cand
                        break
            elif kind == "MISSING_PO":
                cited_ref = None
            elif kind == "DUPLICATE_INVOICE":
                paid.append(inv_no)
            elif kind == "CURRENCY_MISMATCH":
                cur_inv = rng.choice([c for c in ["EUR", "USD", "GBP"] if c != cur_po])

    # ---- ground truth by rule engine (recomputed from the rendered numbers) ----
    present = []
    if cited_ref is None:
        present.append("MISSING_PO")
    if inv_no in paid:
        present.append("DUPLICATE_INVOICE")
    if cur_inv != cur_po:
        present.append("CURRENCY_MISMATCH")
    if abs(inv_price - po_price) / po_price > 0.01 + 1e-9:
        present.append("PRICE_MISMATCH")
    if recv_qty > 0 and abs(inv_qty - recv_qty) / recv_qty > 0.02 + 1e-9:
        present.append("QTY_MISMATCH")
    gold = "NO_EXCEPTION"
    for cat in EXC_PRECEDENCE:
        if cat in present:
            gold = cat
            break

    body = (
        f"VENDOR: {rng.choice(VENDORS)}\n"
        f"PURCHASE ORDER {po_ref}: "
        f"quantity {po_qty}, unit price {po_price:.2f} {cur_po}\n"
        f"GOODS RECEIPT: {recv_qty} units received, condition acceptable\n"
        f"INVOICE {inv_no}: quantity {inv_qty}, unit price {inv_price:.2f} {cur_inv}\n"
        f"PO REFERENCE PRINTED ON INVOICE: {cited_ref if cited_ref else 'none'}\n"
        f"INVOICES ALREADY PAID THIS QUARTER: {', '.join(paid)}\n"
        f"PAYMENT TERMS: net 30"
    )
    return dict(
        case_id=f"EXC-{difficulty}-{idx:04d}",
        task_type="exception_classification",
        difficulty=difficulty,
        instruction=EXC_INSTRUCTION,
        body=body,
        fmt=EXC_FMT,
        gold=gold,
        answer_space=EXC_CATEGORIES,
        meta=dict(po_qty=po_qty, po_price=po_price, recv_qty=recv_qty,
                  inv_qty=inv_qty, inv_price=inv_price, cur_po=cur_po,
                  cur_inv=cur_inv, flags=present),
    )


# ---------------------------------------------------------------------------
# B. field extraction
# ---------------------------------------------------------------------------
EXTRACT_FIELDS = ["invoice_number", "po_reference", "currency", "total_amount", "due_date"]
EXTRACT_INSTRUCTION = (
    "Extract exactly these five fields from the invoice document:\n"
    "- invoice_number: the invoice number exactly as printed.\n"
    "- po_reference: the purchase-order reference exactly as printed.\n"
    "- currency: the three-letter ISO currency code that applies to the amounts.\n"
    "- total_amount: the total amount payable, two decimal places, digits and decimal "
    "point only (no symbol, no thousands separator).\n"
    "- due_date: the payment due date in ISO format YYYY-MM-DD.\n"
    "Every field must be correct; there is no partial credit."
)
EXTRACT_FMT = (
    'Answer with exactly one JSON object on one line:\n'
    '{"invoice_number": "...", "po_reference": "...", "currency": "...", '
    '"total_amount": "...", "due_date": "..."}'
)


def _extract_case(rng: random.Random, difficulty: str, idx: int) -> dict:
    inv_no = f"INV-{rng.randint(300000, 899999)}"
    po_ref = f"PO-{rng.randint(10000, 99999)}"
    cur = rng.choice(["EUR", "USD", "GBP"])
    sym = {"EUR": "EUR", "USD": "USD", "GBP": "GBP"}[cur]
    n_items = rng.choice([2, 3, 3, 4])
    items = []
    subtotal = 0.0
    for _ in range(n_items):
        q = rng.randint(1, 40)
        u = round(rng.uniform(8.0, 320.0), 2)
        a = round(q * u, 2)
        subtotal = round(subtotal + a, 2)
        items.append((f"Item {rng.randint(100, 999)}", q, u, a))
    tax_rate = rng.choice([0.07, 0.19, 0.20, 0.21])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    issue = date(2025, rng.randint(1, 9), rng.randint(1, 27))
    net = rng.choice([14, 30, 45, 60])
    due = issue + timedelta(days=net)
    prev_balance = round(rng.uniform(100.0, 9000.0), 2)

    lines = [f"VENDOR INVOICE", f"{rng.choice(VENDORS)}"]
    if difficulty == "easy":
        lines.append(f"All amounts in {cur}.")
    lines += [
        f"Invoice No.: {inv_no}",
        f"PO Ref: {po_ref}",
        f"Invoice date: {issue.isoformat()}",
        f"Payment terms: Net {net} days",
        f"Vendor Tax ID: {rng.randint(10**8, 10**9 - 1)}",
        "Line items:",
    ]
    for desc, q, u, a in items:
        lines.append(f"  {desc} | qty {q} | unit {u:.2f} | amount {a:.2f}")
    lines.append(f"Subtotal: {subtotal:.2f}")
    lines.append(f"Tax ({int(tax_rate*100)}%): {tax:.2f}")
    lines.append(f"Previous balance: {prev_balance:.2f}")

    if difficulty == "easy":
        lines.append(f"TOTAL DUE: {total:.2f} {sym}")
        lines.append(f"Due date: {due.isoformat()}")
        lines.append(f"Remit to: {rng.randint(100, 999)} Harbour Street")
    elif difficulty == "medium":
        lines.append(f"TOTAL DUE: {total:.2f} {sym}")
        lines.append(f"Remit to: {rng.randint(100, 999)} Harbour Street")
    else:  # hard: total not printed, due date not printed, distractor dates
        stmt = issue - timedelta(days=rng.randint(2, 9))
        lines.append(f"Statement date: {stmt.isoformat()}")
        lines.append(f"Remit to: {rng.randint(100, 999)} Harbour Street")
        lines.append(f"All amounts in {cur}.")

    body = "\n".join(lines)
    gold = {
        "invoice_number": inv_no,
        "po_reference": po_ref,
        "currency": cur,
        "total_amount": f"{total:.2f}",
        "due_date": due.isoformat(),
    }
    return dict(
        case_id=f"EXT-{difficulty}-{idx:04d}",
        task_type="field_extraction",
        difficulty=difficulty,
        instruction=EXTRACT_INSTRUCTION,
        body=body,
        fmt=EXTRACT_FMT,
        gold=gold,
        answer_space=EXTRACT_FIELDS,
        meta=dict(subtotal=subtotal, tax=tax, total=total, net=net,
                  issue=issue.isoformat(), due=due.isoformat(),
                  prev_balance=prev_balance),
    )


# ---------------------------------------------------------------------------
# C. three-way match
# ---------------------------------------------------------------------------
MATCH_POLICY = (
    "MATCH POLICY MP-3\n"
    "1. An invoice may be paid only if all of the following hold: a valid PO exists; "
    "invoiced quantity is within 2% of received quantity; invoiced unit price is within "
    "1% of the PO unit price; the goods were received in acceptable condition; and the "
    "vendor is on the approved list.\n"
    "2. If a quantity or price condition fails but stays inside the correctable band "
    "(quantity deviation of at most 5%, price deviation of at most 3%), the invoice is "
    "placed on HOLD pending a corrected invoice.\n"
    "3. If any condition fails beyond the correctable band, or the goods were damaged, "
    "or the vendor is not approved, the invoice is ESCALATED to procurement review.\n"
    "4. Precedence: ESCALATE > HOLD > PAY."
)
MATCH_INSTRUCTION = "Apply MATCH POLICY MP-3 to the record and decide what to do with the invoice."
MATCH_FMT = "Answer with exactly one line:\nDECISION: <PAY|HOLD|ESCALATE>"


def _match_case(rng: random.Random, difficulty: str, idx: int) -> dict:
    po_qty = rng.choice([50, 80, 100, 120, 200, 400])
    po_price = round(rng.uniform(5.0, 180.0), 2)
    recv_qty = po_qty
    inv_qty = po_qty
    inv_price = po_price
    condition = "acceptable"
    approved = True

    if difficulty == "easy":
        mode = rng.choice(["clean", "blatant", "clean", "blatant"])
        if mode == "blatant":
            which = rng.choice(["price", "qty", "damaged", "unapproved"])
            if which == "price":
                inv_price = round(po_price * (1 + rng.choice([-1, 1]) * rng.uniform(0.18, 0.40)), 2)
            elif which == "qty":
                inv_qty = int(round(recv_qty * (1 + rng.uniform(0.15, 0.35))))
            elif which == "damaged":
                condition = "damaged in transit"
            else:
                approved = False
    elif difficulty == "medium":
        which = rng.choice(["price_hold", "qty_hold", "damaged", "unapproved", "clean"])
        if which == "price_hold":
            inv_price = round(po_price * (1 + rng.choice([-1, 1]) * rng.uniform(0.012, 0.028)), 2)
        elif which == "qty_hold":
            inv_qty = int(round(recv_qty * (1 + rng.uniform(0.022, 0.048))))
        elif which == "damaged":
            condition = "damaged in transit"
            inv_price = round(po_price * rng.uniform(0.997, 1.003), 2)
        elif which == "unapproved":
            approved = False
    else:  # hard: conflicting severities / near-boundary margins
        mode = rng.choice(["hold_vs_escalate", "boundary_in", "boundary_out", "multi"])
        if mode == "hold_vs_escalate":
            inv_price = round(po_price * (1 + rng.uniform(0.012, 0.028)), 2)  # HOLD band
            condition = "damaged in transit"                                   # -> ESCALATE
        elif mode == "boundary_in":
            # just inside the 1% price tolerance -> still PAY
            inv_price = round(po_price * (1 + rng.uniform(0.0075, 0.0095)), 2)
        elif mode == "boundary_out":
            # just outside the 1% price tolerance, inside the 3% band -> HOLD
            inv_price = round(po_price * (1 + rng.uniform(0.0105, 0.0125)), 2)
        else:
            inv_qty = int(round(recv_qty * (1 + rng.uniform(0.024, 0.048))))  # HOLD band
            inv_price = round(po_price * (1 + rng.uniform(0.032, 0.05)), 2)    # ESCALATE
            approved = False

    dq = abs(inv_qty - recv_qty) / recv_qty if recv_qty else 0.0
    dp = abs(inv_price - po_price) / po_price
    if (not approved) or condition != "acceptable" or dq > 0.05 + 1e-9 or dp > 0.03 + 1e-9:
        gold = "ESCALATE"
    elif dq > 0.02 + 1e-9 or dp > 0.01 + 1e-9:
        gold = "HOLD"
    else:
        gold = "PAY"

    body = (
        f"{MATCH_POLICY}\n\n"
        f"RECORD\n"
        f"Vendor: {rng.choice(VENDORS)} "
        f"({'on approved list' if approved else 'NOT on approved list'})\n"
        f"Purchase order: {po_qty} units at {po_price:.2f} each\n"
        f"Goods receipt: {recv_qty} units, condition: {condition}\n"
        f"Invoice: {inv_qty} units at {inv_price:.2f} each"
    )
    return dict(
        case_id=f"MAT-{difficulty}-{idx:04d}",
        task_type="three_way_match",
        difficulty=difficulty,
        instruction=MATCH_INSTRUCTION,
        body=body,
        fmt=MATCH_FMT,
        gold=gold,
        answer_space=["PAY", "HOLD", "ESCALATE"],
        meta=dict(po_qty=po_qty, po_price=po_price, recv_qty=recv_qty,
                  inv_qty=inv_qty, inv_price=inv_price, dq=round(dq, 5),
                  dp=round(dp, 5), condition=condition, approved=approved),
    )


# ---------------------------------------------------------------------------
# D. policy compliance
# ---------------------------------------------------------------------------
POLICY = (
    "VENDOR AND EXPENSE POLICY VP-7\n"
    "C1 Approval threshold: purchases up to 5,000 require one approver; purchases from "
    "5,001 to 25,000 require two approvers; purchases above 25,000 require three "
    "approvers including the Finance Director.\n"
    "C2 Segregation of duties: the requester may not also approve the same purchase.\n"
    "C3 Vendor onboarding: payment may only be made to a vendor whose risk assessment "
    "is dated within the last 24 months of the purchase date.\n"
    "C4 Discount authority: a discount above 15% requires Finance Director sign-off.\n"
    "C5 Expense timing: an expense must be recorded in the period the goods or services "
    "were received; recording more than 30 days after receipt is a violation.\n"
    "C6 Emergency exception: where a documented emergency declaration exists, C1 and C5 "
    "are waived. C2, C3 and C4 are never waived.\n"
    "Precedence when several clauses are violated: C2 > C3 > C4 > C1 > C5."
)
POLICY_INSTRUCTION = (
    "Apply POLICY VP-7 to the purchase record. Decide whether it is compliant, and if "
    "not, name the single highest-precedence clause that is violated."
)
POLICY_FMT = (
    "Answer with exactly one line, either\nVERDICT: COMPLIANT\nor\nVERDICT: VIOLATION <clause id>"
)
PEOPLE = ["A. Okafor", "M. Lindqvist", "R. Bhatt", "S. Moreau", "T. Nakamura",
          "L. Ferreira", "J. Whitfield", "K. Sørensen"]
ROLES = ["Finance Director", "Controller", "Category Manager", "Operations Lead",
         "Procurement Officer"]


def _policy_case(rng: random.Random, difficulty: str, idx: int) -> dict:
    def required_approvers(amt: float) -> int:
        return 1 if amt <= 5000 else (2 if amt <= 25000 else 3)

    requester = rng.choice(PEOPLE)
    others = [p for p in PEOPLE if p != requester]
    purchase_date = date(2025, rng.randint(1, 10), rng.randint(1, 27))
    vendor = rng.choice(VENDORS)
    receipt = purchase_date
    record = purchase_date + timedelta(days=rng.randint(0, 20))
    discount = round(rng.uniform(0.0, 12.0), 1)
    fd_signed = False
    emergency = False
    ra_months_ago = rng.randint(1, 20)

    # ---- compliant baseline: every clause satisfied ----
    amount = round(rng.uniform(600.0, 24000.0), 2)
    need = required_approvers(amount)
    n_appr = need
    approvers = rng.sample(others, 3)
    appr_roles = {a: rng.choice([r for r in ROLES if r != "Finance Director"])
                  for a in approvers}
    if need == 3:
        appr_roles[approvers[0]] = "Finance Director"

    # ---- inject exactly the intended condition(s) ----
    if difficulty == "easy":
        mode = rng.choice(["clean", "clean", "c1", "c2", "c3"])
        if mode == "c1":
            amount = round(rng.uniform(28000.0, 44000.0), 2)
            n_appr = 1  # blatantly too few approvers
        elif mode == "c2":
            approvers[0] = requester
        elif mode == "c3":
            ra_months_ago = rng.randint(30, 44)
    elif difficulty == "medium":
        mode = rng.choice(["c1_fd_missing", "c5_late", "c4_nosign", "c3_edge", "clean"])
        if mode == "c1_fd_missing":
            amount = round(rng.uniform(26000.0, 40000.0), 2)
            n_appr = 3
            appr_roles[approvers[0]] = "Controller"
            appr_roles[approvers[1]] = "Category Manager"
            appr_roles[approvers[2]] = "Operations Lead"  # no Finance Director
        elif mode == "c5_late":
            record = receipt + timedelta(days=rng.randint(33, 70))
        elif mode == "c4_nosign":
            discount = round(rng.uniform(16.0, 28.0), 1)
        elif mode == "c3_edge":
            ra_months_ago = rng.randint(25, 30)
    else:  # hard: waivers and traps
        mode = rng.choice(["emergency_c2", "emergency_compliant", "discount_signed",
                           "c3_waiver_trap", "c2_and_c1"])
        if mode == "emergency_c2":
            emergency = True
            amount = round(rng.uniform(30000.0, 44000.0), 2)
            n_appr = 1                # C1 would be waived
            approvers[0] = requester  # C2 violation survives the waiver
            record = receipt + timedelta(days=45)  # C5 would be waived
        elif mode == "emergency_compliant":
            emergency = True
            amount = round(rng.uniform(30000.0, 44000.0), 2)
            n_appr = 1
            record = receipt + timedelta(days=44)
        elif mode == "discount_signed":
            discount = round(rng.uniform(17.0, 26.0), 1)
            fd_signed = True
            appr_roles[approvers[0]] = "Finance Director"
            amount = round(rng.uniform(7000.0, 22000.0), 2)
            n_appr = 2
        elif mode == "c3_waiver_trap":
            emergency = True
            ra_months_ago = rng.randint(28, 38)
            amount = round(rng.uniform(6000.0, 20000.0), 2)
            n_appr = 2
        else:  # c2_and_c1
            amount = round(rng.uniform(26000.0, 42000.0), 2)
            n_appr = 2
            approvers[0] = requester

    risk_date = purchase_date - timedelta(days=int(ra_months_ago * 30.4))

    # ---- rule engine ----
    viol = {}
    if requester in approvers[:n_appr]:
        viol["C2"] = True
    months_since = (purchase_date - risk_date).days / 30.4
    if months_since > 24:
        viol["C3"] = True
    if discount > 15.0 and not fd_signed:
        viol["C4"] = True
    if not emergency:
        need = required_approvers(amount)
        fd_present = any(appr_roles.get(a) == "Finance Director" for a in approvers[:n_appr])
        if n_appr < need or (need == 3 and not fd_present):
            viol["C1"] = True
        if (record - receipt).days > 30:
            viol["C5"] = True
    gold = "COMPLIANT"
    for c in ["C2", "C3", "C4", "C1", "C5"]:
        if viol.get(c):
            gold = c
            break

    appr_txt = "; ".join(
        f"{a} ({appr_roles.get(a, 'Controller')})" for a in approvers[:n_appr]
    )
    body = (
        f"{POLICY}\n\n"
        f"RECORD\n"
        f"Requester: {requester}\n"
        f"Vendor: {vendor}\n"
        f"Purchase amount: {amount:.2f}\n"
        f"Approvers recorded: {appr_txt}\n"
        f"Vendor risk assessment date: {risk_date.isoformat()}\n"
        f"Discount granted: {discount:.1f}%\n"
        f"Finance Director sign-off on discount: {'yes' if fd_signed else 'no'}\n"
        f"Goods received: {receipt.isoformat()}\n"
        f"Expense recorded: {record.isoformat()}\n"
        f"Emergency declaration on file: {'yes' if emergency else 'no'}"
    )
    return dict(
        case_id=f"POL-{difficulty}-{idx:04d}",
        task_type="policy_compliance",
        difficulty=difficulty,
        instruction=POLICY_INSTRUCTION,
        body=body,
        fmt=POLICY_FMT,
        gold=gold,
        answer_space=["COMPLIANT", "C1", "C2", "C3", "C4", "C5"],
        meta=dict(amount=amount, n_appr=n_appr, requester=requester,
                  risk_months=round(months_since, 1), discount=discount,
                  fd_signed=fd_signed, days_to_record=(record - receipt).days,
                  emergency=emergency, viol=sorted(viol)),
    )


GENERATORS = {
    "exception_classification": _exc_case,
    "field_extraction": _extract_case,
    "three_way_match": _match_case,
    "policy_compliance": _policy_case,
}


def build_battery(n_per_cell: int = 34, seed: int = 20251027) -> list[dict]:
    cases: list[dict] = []
    for ti, tt in enumerate(TASK_TYPES):
        for si, stratum in enumerate(STRATA):
            rng = random.Random(seed + 1000 * ti + 100 * si)
            for k in range(n_per_cell):
                c = GENERATORS[tt](rng, stratum, k)
                c["prompt_paraphrases"] = [
                    render(p, c["instruction"], c["body"], c["fmt"])
                    for p in range(N_PARAPHRASE)
                ]
                cases.append(c)
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=34, help="cases per (task type, stratum) cell")
    ap.add_argument("--out", default=str(ROOT / "tasks" / "battery.jsonl"))
    args = ap.parse_args()
    cases = build_battery(args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")
    from collections import Counter
    print(f"wrote {len(cases)} cases -> {out}")
    print("by task_type:", dict(Counter(c["task_type"] for c in cases)))
    print("by difficulty:", dict(Counter(c["difficulty"] for c in cases)))
    print("gold distribution:", dict(Counter(
        c["gold"] if isinstance(c["gold"], str) else "STRUCTURED" for c in cases)))


if __name__ == "__main__":
    main()
