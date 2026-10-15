# Invention disclosure drafts

Two disclosures, filled from what this repository measures rather than from
recollection. Every figure in them is reproducible from the code and the
recorded state; none was rounded for effect.

| File | For |
| --- | --- |
| `IDF-B-Patent-SLA-Scheduler-FILLED.docx` | Patent — the SLA-constrained scheduler |
| `IDF-Design-Registration-Interface-FILLED.docx` | Design registration — the interface |
| `IDF-B-template-blank.pdf`, `IDF-Design-template-blank.pdf` | The blank forms, for comparison |

## What the patent disclosure claims, and what it does not

The principal claim is the scheduler: a per-hardware-class execution cost
maintained as an affine function of input duration, fitted online, driving a
joint choice of hardware class and model fidelity against a wall-clock
deadline. The models themselves — Whisper, YOLO, CLIP, the language models —
are used as interchangeable components and are explicitly not claimed.

Section 3(k) of the Patents Act excludes a computer programme per se, an
algorithm and a mathematical method. The disclosure is therefore written
around the technical effect on resource utilisation, which is what is
measured in item 8, rather than around the computation.

## Before this is submitted

- [ ] **Prior-art search** through the VIT IPR Cell. No patent numbers are
      cited in item 3 because none have been verified, and inventing one in a
      legal document is not an option. The search categories the inventors
      believe matter are listed there instead.
- [ ] **Three drawings** for item 7: the architecture, the measured cost
      against input length with both models plotted, and the routing flowchart.
- [ ] **Seven screen representations** for item 3 of the design form.
- [ ] **Contributor details** — name, address, email, phone — in item 7 of the
      design form, and signatures in item 8.
- [ ] **Review date** in item 5 of the design form.
- [ ] **A decision on the public endpoint.** The running system is currently
      reachable over the internet through a tunnelled address with no access
      control on its landing page. Both forms record this as a fact and ask
      the Cell whether it amounts to prior publication. Until that is
      answered, the prudent course is to put the endpoint behind sign-in or
      take it down.

## Reproducing the figures

```bash
pytest tests/                                     # 188 tests
curl localhost:8000/api/scheduler/sla             # deadline attainment
python -m ml.evaluate --checkpoint ml/checkpoints/importance.pt --baselines
```
