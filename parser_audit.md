# Parser extraction audit — stratified sample of 50 grid responses

## Reproducibility

- **Random seed:** `20261030`
- **Source:** `common-cause-fleet/results/grid.jsonl` (4,896 records; fixed)
- **Algorithm:** iterate `grid.jsonl` in file order; group by `(model, difficulty)` preserving first-seen order; `random.shuffle` each group with the shared RNG; take up to 6 per group; concatenate in group order; `random.shuffle`; take the first 50. Because the source is fixed, the manifest below reproduces exactly.

## Sample manifest (50 records, in sampling order)

| # | case_id | model | difficulty | slot | task_type | stored parse | in raw | re-derived | match | pipeline correct |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | EXC-easy-0033 | M2 | easy | 3 | exception_classification | DUPLICATE_INVOICE | yes | duplicate_invoice | yes | True |
| 2 | MAT-hard-0028 | M2 | hard | 1 | three_way_match | HOLD | yes | hold | yes | False |
| 3 | POL-medium-0015 | M1 | medium | 1 | policy_compliance | C4 | yes | c4 | yes | False |
| 4 | MAT-easy-0012 | M2 | easy | 3 | three_way_match | ESCALATE | yes | escalate | yes | True |
| 5 | EXC-medium-0014 | M2 | medium | 3 | exception_classification | QTY_MISMATCH | yes | qty_mismatch | yes | False |
| 6 | POL-hard-0020 | M1 | hard | 4 | policy_compliance | C2 | yes | c2 | yes | True |
| 7 | MAT-easy-0026 | M3 | easy | 3 | three_way_match | PAY | yes | pay | yes | True |
| 8 | EXT-hard-0020 | M1 | hard | 2 | field_extraction | {'invoice_number': 'INV-405070', 'po_reference | yes | {'invoice_number': 'INV-405070', 'po_ref | yes | True |
| 9 | POL-hard-0014 | M1 | hard | 4 | policy_compliance | C2 | yes | c2 | yes | True |
| 10 | EXC-hard-0018 | M3 | hard | 2 | exception_classification | MISSING_PO | yes | missing_po | yes | True |
| 11 | MAT-medium-0010 | M2 | medium | 1 | three_way_match | ESCALATE | yes | escalate | yes | True |
| 12 | POL-easy-0008 | M3 | easy | 2 | policy_compliance | C4 | yes | c4 | yes | False |
| 13 | POL-medium-0024 | M1 | medium | 3 | policy_compliance | COMPLIANT | yes | compliant | yes | True |
| 14 | EXC-easy-0025 | M1 | easy | 2 | exception_classification | PRICE_MISMATCH | yes | price_mismatch | yes | True |
| 15 | POL-medium-0024 | M1 | medium | 1 | policy_compliance | C1 | yes | c1 | yes | False |
| 16 | MAT-easy-0007 | M2 | easy | 1 | three_way_match | PAY | yes | pay | yes | True |
| 17 | EXC-easy-0027 | M3 | easy | 2 | exception_classification | NO_EXCEPTION | yes | no_exception | yes | False |
| 18 | POL-medium-0030 | M2 | medium | 1 | policy_compliance | C4 | yes | c4 | yes | True |
| 19 | POL-medium-0001 | M2 | medium | 3 | policy_compliance | C3 | yes | c3 | yes | False |
| 20 | EXC-hard-0014 | M3 | hard | 2 | exception_classification | MISSING_PO | yes | missing_po | yes | True |
| 21 | EXT-hard-0030 | M1 | hard | 1 | field_extraction | {'invoice_number': 'INV-340925', 'po_reference | yes | {'invoice_number': 'INV-340925', 'po_ref | yes | True |
| 22 | MAT-easy-0018 | M1 | easy | 2 | three_way_match | PAY | yes | pay | yes | True |
| 23 | EXC-medium-0018 | M3 | medium | 0 | exception_classification | MISSING_PO | yes | missing_po | yes | False |
| 24 | EXT-hard-0011 | M1 | hard | 3 | field_extraction | {'invoice_number': 'INV-704267', 'po_reference | yes | {'invoice_number': 'INV-704267', 'po_ref | yes | False |
| 25 | EXC-hard-0017 | M3 | hard | 2 | exception_classification | MISSING_PO | yes | missing_po | yes | True |
| 26 | EXT-easy-0012 | M1 | easy | 0 | field_extraction | {'invoice_number': 'INV-486575', 'po_reference | yes | {'invoice_number': 'INV-486575', 'po_ref | yes | True |
| 27 | EXC-medium-0001 | M1 | medium | 1 | exception_classification | QTY_MISMATCH | yes | qty_mismatch | yes | False |
| 28 | EXC-medium-0000 | M3 | medium | 2 | exception_classification | NO_EXCEPTION | yes | no_exception | yes | False |
| 29 | EXC-easy-0026 | M3 | easy | 0 | exception_classification | MISSING_PO | yes | missing_po | yes | True |
| 30 | EXT-easy-0004 | M2 | easy | 1 | field_extraction | {'invoice_number': 'INV-783178', 'po_reference | yes | {'invoice_number': 'INV-783178', 'po_ref | yes | True |
| 31 | EXC-hard-0029 | M2 | hard | 1 | exception_classification | DUPLICATE_INVOICE | yes | duplicate_invoice | yes | True |
| 32 | EXC-easy-0031 | M3 | easy | 3 | exception_classification | MISSING_PO | yes | missing_po | yes | True |
| 33 | MAT-hard-0012 | M2 | hard | 1 | three_way_match | ESCALATE | yes | escalate | yes | True |
| 34 | EXC-hard-0027 | M3 | hard | 3 | exception_classification | MISSING_PO | yes | missing_po | yes | True |
| 35 | MAT-medium-0016 | M3 | medium | 1 | three_way_match | HOLD | yes | hold | yes | False |
| 36 | POL-easy-0030 | M1 | easy | 4 | policy_compliance | C1 | yes | c1 | yes | False |
| 37 | EXC-easy-0004 | M1 | easy | 1 | exception_classification | DUPLICATE_INVOICE | yes | duplicate_invoice | yes | True |
| 38 | POL-hard-0029 | M2 | hard | 3 | policy_compliance | C2 | yes | c2 | yes | True |
| 39 | POL-hard-0009 | M1 | hard | 3 | policy_compliance | C1 | yes | c1 | yes | False |
| 40 | EXC-medium-0029 | M3 | medium | 2 | exception_classification | PRICE_MISMATCH | yes | price_mismatch | yes | False |
| 41 | EXC-easy-0013 | M3 | easy | 0 | exception_classification | NO_EXCEPTION | yes | no_exception | yes | True |
| 42 | POL-easy-0010 | M1 | easy | 2 | policy_compliance | C3 | yes | c3 | yes | True |
| 43 | POL-hard-0015 | M2 | hard | 1 | policy_compliance | C3 | yes | c3 | yes | True |
| 44 | EXC-medium-0022 | M3 | medium | 3 | exception_classification | CURRENCY_MISMATCH | yes | currency_mismatch | yes | True |
| 45 | EXC-medium-0012 | M2 | medium | 3 | exception_classification | QTY_MISMATCH | yes | qty_mismatch | yes | False |
| 46 | POL-medium-0008 | M1 | medium | 0 | policy_compliance | C5 | yes | c5 | yes | True |
| 47 | EXC-medium-0032 | M3 | medium | 2 | exception_classification | NO_EXCEPTION | yes | no_exception | yes | False |
| 48 | EXC-hard-0004 | M2 | hard | 1 | exception_classification | MISSING_PO | yes | missing_po | yes | True |
| 49 | MAT-easy-0033 | M2 | easy | 1 | three_way_match | ESCALATE | yes | escalate | yes | True |
| 50 | EXT-medium-0016 | M1 | medium | 4 | field_extraction | {'invoice_number': 'INV-711752', 'po_reference | yes | {'invoice_number': 'INV-711752', 'po_ref | yes | True |

## Agreement by model × difficulty

| Model | Difficulty | Sampled | Agreement |
|---|---|---|---|
| M1 | easy | 6 | 6 |
| M1 | hard | 6 | 6 |
| M1 | medium | 6 | 6 |
| M2 | easy | 5 | 5 |
| M2 | hard | 6 | 6 |
| M2 | medium | 5 | 5 |
| M3 | easy | 6 | 6 |
| M3 | hard | 4 | 4 |
| M3 | medium | 6 | 6 |
| All | -- | 50 | 50 |