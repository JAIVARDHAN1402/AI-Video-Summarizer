# Prior-art assessment — SLA-constrained cost-aware scheduler

Prepared 2026-10-16, before filing, from a web and Google Patents search.

## What this is, and what it is not

This is a **freedom-to-claim sanity check**, not a patent search. A real search
runs CPC classification queries (G06F 9/4881, G06F 9/50, H04N 21/2343) across
Espacenet, USPTO full-text and INPASS, reads the claims of every family member,
and is signed by someone who can be held to it. Only the VIT IPR Cell's agent
can do that.

What follows is the closest art I could actually read, mapped element by
element against the invention. Where I say "anticipated", I read the document
and am quoting what it discloses.

## The closest art found

| Reference | What it actually discloses |
|---|---|
| **US 2014/0229221 A1** | Estimates job duration from **prior executions** of similar jobs. Estimates that duration across **multiple instance pools of differing configuration**. Selects the **lowest-cost pool that completes before a need-by time**. Migrates to a faster pool if the job falls behind. States the size-to-duration relationship "may be determined to be linear". Domain: cloud batch compute. |
| **US 2015/0006226 A1** | Transcode-time estimate as a function of file size or duration plus transcode type. **No learning, no feedback.** Deadline handled by reordering and penalties. |
| **INFaaS** (USENIX ATC '21) | Selects **model variant *and* hardware architecture jointly** under latency / accuracy / cost SLOs; autoscaler minimises hardware cost subject to the SLO. Per-query inference serving. |
| **US 12,469,288** | Scheduler with a **latency model** choosing among **execution branches** of a video detector, including the cost of switching branches. |
| **US 10,685,235 / 11,354,902** | Querying video with reduced latency and cost; trades compressed-model accuracy against ingest cost. |
| **FastTTPS**, Deneke et al. | Transcoding-time prediction by **linear regression** on video characteristics (FastTTPS reports R² = 0.988), used to drive scheduling. |

## Element-by-element verdict

| Element of the invention | Prior art | Verdict |
|---|---|---|
| A. Predict total execution time for a media job from input length | FastTTPS, Deneke, US2015/0006226, US2014/0229221 | **Anticipated** |
| B. Affine form: `fixed + rate x minutes`, fitted from past runs | Linear regression with an intercept is textbook; US2014/0229221 contemplates a linear relationship | **Anticipated / obvious** |
| C. A separate model per hardware class | US2014/0229221 estimates across multiple instance pools | **Anticipated** |
| D. Pick the **cheapest** class that meets the deadline, not the fastest | US2014/0229221, claim-level | **Squarely anticipated** |
| E. Bounded exploration — probe unmeasured or thin classes only where the deadline cannot break | Not found in the media-scheduling art; it is recognisable as bandit exploration under a constraint | **Arguable, but a known technique** |
| F. Asymmetric update (overruns believed fast, improvements slowly) | **Not found anywhere** | **Appears novel** |
| G. Joint hardware-class **and** model-fidelity selection under one deadline | INFaaS; US 12,469,288 | **Largely anticipated** |
| H. Declare infeasibility and report expected overrun **before** the job starts | Not found explicitly | Minor, appears novel |
| I. Online update from five running sums (constant work per completion) | Recursive least squares / incremental LS is textbook | **Obvious** |

## The honest conclusion

**The broad idea is not patentable.** "Learn job duration from past runs, then
pick the cheapest machine that still meets the deadline" is US 2014/0229221 A1.
That is the core of the scheduler as currently described in the IDF, and it is
already published prior art. A claim written at that level will be rejected on
novelty.

**What may survive is a narrow combination claim**, something like:

> ...wherein the execution-cost model for each hardware class comprises an
> explicit fixed-overhead term and a per-input-minute rate term, updated after
> each completed job in constant work from running sums; wherein the fit is
> used only after at least N completions spanning at least a 2x range of input
> lengths, and a coarser estimator is used otherwise; wherein observed
> overruns revise the model faster than observed improvements; and wherein
> exploratory assignment to an unmeasured class is permitted only when the
> predicted completion still satisfies the deadline.

That is elements B + E + F + I + the credibility gate, as a combination — not
the scheduling idea itself. It is defensible but **narrow**: a competitor avoids
it by using a non-affine fit, or a symmetric update, or no exploration gate.

**Second hurdle: Section 3(k).** A scheduling method implemented in software is
a "computer programme per se" unless a technical effect is demonstrated. The
operative guidance is now the **CRI Guidelines 2025** (the 2017 version is
superseded), and after *Ferid Allani* (Delhi HC, 2019) a demonstrable technical
effect is the test. The project's strongest asset here is the **measured
result**: 68.0% SLA attainment against 8.0% for the fixed-assignment baseline,
on 25 non-resumed runs. That is a concrete technical effect — better deadline
attainment on heterogeneous hardware — and it must be in the specification with
the measurement method stated.

**Known weakness in the cost model.** A 2023 transcoding study finds execution
time also depends on target bitrate, not only on duration. A claim that models
cost from input minutes alone is vulnerable to the argument that it is an
incomplete model. Either widen the feature to include a second term, or confine
the claim to a fixed output configuration.

## Recommendation

1. **Do not file the broad claim.** Take US 2014/0229221 A1 to the IPR Cell
   yourself and say "this is the closest art I found" — it is far better that
   you found it than that the examiner does.
2. **File the narrow combination**, if the agent agrees it clears the art. The
   fallback-and-gating behaviour and the asymmetric update are the parts no
   reference showed.
3. **The Design Registration on the interface is the stronger of the two
   filings** and is much more likely to be granted. Do not let the patent
   question delay it.
4. **Publish the measurement work regardless.** The 68% vs 8% comparison and
   the TVSum / SumMe baselines are publishable on their own merit, and
   publication is not blocked by the patent position — but file first, because
   publishing before filing destroys novelty.
5. Ask the IPR Cell to run the formal search under G06F 9/4881 and G06F 9/50
   before any money is spent on drafting.

## Sources read

- US 2014/0229221 A1 — Google Patents
- US 2015/0006226 A1 — Google Patents
- US 12,469,288 — USPTO full-text
- US 10,685,235 and US 11,354,902 — USPTO full-text
- INFaaS, USENIX ATC 2021
- FastTTPS; Deneke et al., transcoding-time prediction
- CRI Guidelines 2025, Indian Patent Office
