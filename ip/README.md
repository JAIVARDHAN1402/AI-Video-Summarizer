# Intellectual property

Disclosure drafts for the VIT IPR Cell, with the blank templates they were
written onto.

| File | What it is |
| --- | --- |
| `IDF-B-Patent-SLA-Scheduler-FILLED.docx` | Patent disclosure for the scheduler |
| `IDF-Design-Registration-Interface-FILLED.docx` | Design disclosure for the interface |
| `IDF-B-template-blank.pdf` | The blank IDF-B template |
| `IDF-Design-template-blank.pdf` | The blank design-registration template |

## What is claimed

The patent disclosure claims the **scheduling method**, not the pipeline. The
pipeline is assembled from published models (Whisper, YOLO, CLIP, EasyOCR,
Gemini, MiniLM, FAISS) and is explicitly disclaimed in item 9.

The two elements put forward as novel:

1. Choosing the **hardware class and the model-fidelity configuration
   together** against one wall-clock deadline, the configuration being sized
   to the measured speed of the class that was chosen.
2. Learning each class's cost as an **affine function of input length** —
   fixed setup plus a per-unit rate — online, by least squares over five
   running sums, rather than as a single rate.

Both live in `server/scheduler.py`; `core/perf.py` holds the per-stage budget
planner the router hands work to.

## Evidence the disclosure cites

Every figure in item 8 is a measurement from this repository, reproducible
from the recorded data:

- `sla_observations` in `server_state.db` — 28 rows, each holding predicted
  time, actual time, the deadline and whether it was met, alongside the
  fixed-configuration baseline recorded for the same job.
- `scheduler_costs` — the learned per-class cost models.
- `tests/test_fleet_scheduler.py` — tests that the affine fit recovers known
  parameters and that an unsupported fit is refused.

The headline result: an 11 h 40 min recording was processed in 90.2 minutes.
A single-rate model predicted 316 minutes for it; the affine model predicts
94.7. That gap is the technical problem the disclosure is built around.

## Outstanding before filing

- [ ] **Patent prior-art search** (Indian and international). Not done. No
      patent numbers appear in the disclosure because none have been
      verified. The three categories to search are named under item 3.
- [ ] Confirm the dates of the internal review presentations (design form,
      item 4).
- [ ] Capture the seven screen representations listed in the design form,
      item 3.
- [ ] Complete the address, email and phone fields.
- [ ] Prepare the three drawings named in item 7.

## Disclosure timeline, as known

- 20 Aug 2026 — first commit; repository public.
- 6 Sep 2026 — repository made private, and still private.
- The three-panel interface described in the design disclosure was built
  after that date.
- No public deployment has taken place. Indian law provides no general grace
  period for prior disclosure, so a public deployment should wait on the IPR
  Cell's advice.
