"""Chapters 1 to 9 and Appendix A of the Project-I report.

Numbers come from docs/report_data.py, which reads them out of the running
system, so this file states no measurement of its own.
"""
from docx.enum.text import WD_ALIGN_PARAGRAPH

CENTER = WD_ALIGN_PARAGRAPH.CENTER


def pct(x):
    return f"{x * 100:.1f}%"


def build(m, doc, F):
    P, J, B = m.p, m.justified, m.bullet
    CH, SEC, FIGURE, TABLE, CODE = (m.chapter, m.section, m.figure, m.table,
                                    m.code_block)
    s = F["sched"]
    run = F["run"]
    ml = F["ml"]

    # ================================================================ ABSTRACT
    doc.add_page_break()
    P(doc, "ABSTRACT", size=16, bold=True, align=CENTER, space_after=16)
    J(doc,
      "A recorded lecture is cheap to produce and expensive to watch. A "
      "student who needs one answer from a three-hour tutorial has no way to "
      "reach it except to sit through the tutorial. Automatic video "
      "summarisation addresses this, but the systems that do it share a "
      "practical flaw: they run a fixed set of models and finish whenever "
      "they finish. On a long lecture that can mean hours of processing, and "
      "the user is given no estimate before committing to the wait.")
    J(doc,
      "This project treats the deadline as an input rather than an outcome. "
      "It builds a seven-stage multimodal pipeline — ingest, transcribe, "
      "vision, fuse, understand, index and highlight — and places in "
      "front of it a scheduler that keeps, for every available hardware "
      "class, an execution-cost model of the affine form "
      "c(x) = a + b·x, where x is the length of the input in minutes. "
      "The model is fitted online from completed jobs in constant work per "
      "job, using five running sums, and it is used before a job starts to "
      "choose the cheapest hardware class predicted to finish inside the "
      "budget. Where no class can, the shortfall is reported before the run "
      "begins instead of discovered after it.")
    J(doc,
      f"Over {s['observations']} recorded runs the scheduler met its budget "
      f"on {pct(s['met_rate'])} of jobs, against {pct(s['fixed_config_met'])} "
      "for a fixed, highest-fidelity configuration. The comparison is "
      "qualified in Chapter 7: counterfactuals computed from the measured "
      "per-minute costs suggest that always choosing the largest class would "
      f"have met the budget on {pct(s['always'].get('cpu-16core', 0))} of the "
      "same jobs, so the scheduler's contribution is properly described as "
      "comparable attainment at lower cost rather than higher attainment. A "
      f"{run['minutes']}-minute lecture completes in {run['elapsed']} s, "
      f"about {run['speedup']}× faster than real time.")
    J(doc,
      "The summarising model itself is evaluated against human agreement "
      "rather than F1 alone, following the protocol of Otani et al. Under "
      f"{ml['folds']}-fold cross-validation over {ml['n_videos']} videos it "
      f"reaches F1 {ml['per_dataset']['tvsum']['f1']:.3f} on TVSum, against a "
      f"human agreement of {ml['baselines']['tvsum']['human']:.3f} and a "
      f"random baseline of {ml['baselines']['tvsum']['random']:.3f}. On SumMe "
      "its rank correlation is close to chance, and the report says so.")
    J(doc,
      f"The system is implemented in {F['repo']['lines']:,} lines across "
      f"{F['repo']['files']} source files, is covered by {F['tests']} "
      "automated tests, and is deployed with authentication, per-tier "
      "allowances and payment integration working against live provider APIs.")

    # =============================================================== CHAPTER 1
    CH(doc, 1, "Introduction")
    SEC(doc, "1.1 Background")
    J(doc,
      "Lecture capture, recorded tutorials and conference talks have made "
      "long-form educational video abundant. The economics of consuming it "
      "have not improved at the same rate: a video is a linear medium, and "
      "the only index into it most platforms provide is a scrub bar. A "
      "learner who wants to know whether a three-hour SQL tutorial covers "
      "window functions, and at what minute, must either watch it or guess.")
    J(doc,
      "Automatic video summarisation has been studied for over a decade. "
      "Classical work segments a video into shots and scores each for "
      "importance, producing a shorter video. More recent systems transcribe "
      "the audio and pass the transcript to a language model, producing text. "
      "Both lines of work optimise for the quality of the summary. Neither "
      "treats the time taken to produce it as a quantity the user is "
      "entitled to know in advance.")
    J(doc,
      "That omission matters in practice. The stages of a multimodal "
      "pipeline do not cost the same: speech recognition and vision dominate, "
      "and both scale with the length of the input. On ordinary hardware a "
      "long lecture can take longer to process than to watch, which defeats "
      "the purpose. The hardware a job runs on, and the fidelity of the "
      "models it uses, are therefore not implementation details but the main "
      "levers available for delivering a result in a usable time.")

    SEC(doc, "1.2 Motivations")
    J(doc, "Three observations motivated this work.")
    B(doc, "A result that arrives after an unknown delay is hard to plan "
           "around. A user deciding whether to submit a three-hour video "
           "needs an estimate before committing, not a progress bar "
           "afterwards.")
    B(doc, "The cost of a job is predictable in a way that is easy to "
           "exploit. Execution time on a given machine is close to affine in "
           "the length of the input: a fixed overhead for setting up models, "
           "plus a rate per minute of material. If that relationship is "
           "learned, the finishing time can be predicted before the job "
           "starts.")
    B(doc, "Choosing the fastest available machine wastes the deadline. If a "
           "smaller and cheaper machine also finishes in time, the headroom "
           "bought by the larger one is paid for and never used. The "
           "interesting objective is the cheapest configuration that still "
           "keeps the promise.")

    SEC(doc, "1.3 Scope of the Project")
    J(doc,
      "The project delivers a working web application and the scheduler "
      "behind it. In scope: ingestion from a link or an upload; transcription "
      "with automatic language detection; scene detection, keyframe "
      "extraction and on-screen text recognition; alignment of speech to "
      "scene; a language-model pass that produces a chaptered explanation, a "
      "quiz, flashcards, a mind map and a practice set; a searchable index "
      "that answers questions with citations back to the moment in the video; "
      "a highlight reel; authentication and per-tier allowances; and the "
      "scheduler that predicts, routes and calibrates.")
    J(doc,
      "Out of scope: training new speech or vision models — these are "
      "used as interchangeable components; real-time or live-stream "
      "processing; and a multi-tenant production deployment with its own "
      "hardware fleet. The scheduler is designed for heterogeneous worker "
      "classes, and is exercised here across classes emulated on one machine, "
      "which Chapter 7 states as a limitation of the evidence rather than of "
      "the design.")

    # =============================================================== CHAPTER 2
    CH(doc, 2, "Project Description and Goals")
    SEC(doc, "2.1 Literature Review")
    J(doc,
      "Each of the works below was read in full. They fall into three "
      "groups: summarisation quality, runtime prediction, and deadline-aware "
      "scheduling. The project sits at the intersection of the second and "
      "third, and adopts the evaluation protocol of the first.")
    TABLE(doc, "Summary of the literature reviewed", [
        ["Source", "Contribution", "Limitation for this problem"],
        ["Potapov et al., ECCV 2014 (KTS)",
         "Kernel temporal segmentation of a video into shots, with "
         "category-specific importance scoring.",
         "Segmentation and scoring only; no notion of a processing budget."],
        ["Otani et al., CVPR 2019",
         "Shows that F1 on TVSum and SumMe can exceed human agreement while "
         "ranking segments no better than chance; proposes rank correlation.",
         "An evaluation critique, not a system. Its protocol is adopted in "
         "Chapter 7."],
        ["Radford et al., ICML 2023 (Whisper)",
         "Weakly supervised speech recognition robust across languages and "
         "accents.",
         "A component. Its cost scales with audio length, which is what the "
         "scheduler must predict."],
        ["Deneke et al., ICME 2014; FastTTPS",
         "Predict video transcoding time by linear regression on video "
         "characteristics; FastTTPS reports R² = 0.988.",
         "Prediction is not used to choose between machines of differing "
         "cost under a deadline."],
        ["US 2014/0229221 A1",
         "Estimates job duration from prior executions across several "
         "instance pools and selects the lowest-cost pool that finishes "
         "before a need-by time.",
         "Cloud batch compute. Duration is modelled as proportional to input "
         "size, with no explicit fixed-overhead term, and model fidelity is "
         "never traded to keep the deadline."],
        ["Romero et al., USENIX ATC 2021 (INFaaS)",
         "Selects model variant and hardware architecture jointly under "
         "latency, accuracy and cost objectives.",
         "Per-query inference serving, where cost does not scale with the "
         "length of the input; decisions are made in milliseconds."],
    ], widths=[1.5, 2.3, 2.3], font=9.5)

    SEC(doc, "2.2 Research Gap")
    J(doc,
      "Runtime prediction and deadline-aware placement both exist, but in "
      "different literatures and for different workloads. The transcoding "
      "work predicts duration and stops there. The cloud-scheduling work "
      "places jobs against a deadline but models duration as proportional to "
      "input size, without a fixed-overhead term, and does not consider "
      "lowering the fidelity of the computation as an alternative to missing "
      "the deadline. The inference-serving work chooses model and hardware "
      "together, but for single queries whose cost does not grow with an "
      "input length. The summarisation literature considers neither.")
    J(doc,
      "The gap this project addresses is a single system that maintains a "
      "per-hardware-class execution-cost model with an explicit fixed term "
      "and a per-input-minute rate, fits it online from completed jobs, "
      "routes each job to the cheapest class predicted to meet its deadline, "
      "explores unmeasured classes only where the deadline cannot be broken, "
      "and reports an unreachable deadline before the job starts. A "
      "prior-art search carried out before filing is reported in Section 7.6; "
      "it found that the broad form of this claim appears anticipated, and "
      "that what remains distinct is narrower than it first appeared. That "
      "finding is stated here rather than omitted.")

    SEC(doc, "2.3 Objectives")
    for o in [
        "Implement the seven-stage pipeline end to end, from a link or an "
        "upload to a chaptered explanation, transcript, quiz, mind map and "
        "practice set.",
        "Maintain an execution-cost model per hardware class, of the form "
        "c(x) = a + b·x, fitted online from completed jobs in constant "
        "work per job.",
        "Route each submitted job to the cheapest class predicted to meet its "
        "wall-clock budget, and declare an unreachable budget before the job "
        "starts.",
        "Explore classes that are unmeasured or thinly measured, but only "
        "where the exploration itself is predicted to keep the deadline.",
        "Resume a part-completed run from cached stage artefacts rather than "
        "recomputing it, and exclude such runs from calibration.",
        "Evaluate the summarising model against human agreement on TVSum and "
        "SumMe, reporting rank correlation beside F1.",
        "Deliver the whole as an authenticated web application with per-tier "
        "allowances.",
    ]:
        B(doc, o)

    SEC(doc, "2.4 Problem Statement")
    J(doc,
      "Let a job be a video of length x minutes submitted with a wall-clock "
      "budget T seconds. Let K be the set of available hardware classes, each "
      "class k having a price rate pₖ and an execution-cost model "
      "cₖ(x) = aₖ + bₖ·x estimated from that class's own "
      "completed jobs. The routing problem is:")
    P(doc, "minimise  pₖ · cₖ(x)   over k ∈ K,   "
           "subject to  cₖ(x) ≤ T", align=CENTER, bold=True,
      space_after=10)
    J(doc,
      "with two departures from the textbook form. First, the coefficients "
      "aₖ and bₖ are not given; they are estimated from the class's "
      "own history, and the estimate must be refused as not yet credible when "
      "the history is too short or too narrow in the range of lengths seen. "
      "Second, when the feasible set is empty the problem has no solution in "
      "the stated form, and the system must still act: it selects the fastest "
      "class, which bounds how far the miss goes, and reports the expected "
      "overrun before the job starts.")

    SEC(doc, "2.5 Project Plan")
    TABLE(doc, "Phases of the project", [
        ["Phase", "Work", "Status"],
        ["1", "Pipeline stages M1–M4: ingest, transcribe, vision, fuse",
         "Complete"],
        ["2", "Understanding stage M5′, windowed for long inputs",
         "Complete"],
        ["3", "Index and highlight, M6 and M7; importance model trained and "
              "cross-validated", "Complete"],
        ["4", "Scheduler: per-class cost models, routing, calibration",
         "Complete"],
        ["5", "Web application, authentication, allowances, payments",
         "Complete"],
        ["6", "Evaluation against human agreement; SLA measurement",
         "Complete"],
        ["7", "Invention disclosure and prior-art search",
         "Disclosures drafted; filing pending the IPR Cell's formal search"],
        ["8", "Measured A/B against a fixed largest-class assignment",
         "Not started; identified in Chapter 8 as the next step"],
    ], widths=[0.6, 3.7, 1.8], font=10)

    # =============================================================== CHAPTER 3
    CH(doc, 3, "Technical Specification")
    SEC(doc, "3.1 Requirements")
    SEC(doc, "3.1.1 Functional Requirements", level=3)
    for r in [
        "FR1 — Accept a video as a link or an uploaded file, and show a "
        "predicted finishing time before processing begins.",
        "FR2 — Transcribe the audio, detecting the language "
        "automatically rather than assuming English.",
        "FR3 — Detect scenes, extract keyframes and read on-screen text.",
        "FR4 — Align each transcribed sentence to the scene it was "
        "spoken over.",
        "FR5 — Produce a chaptered explanation with timestamps, a scene "
        "grid, a transcript, a quiz, flashcards, a mind map, a slide deck, an "
        "audio overview and an infographic.",
        "FR6 — Extract every problem solved in the video and restate it "
        "in any dialect of the same family: MySQL, PostgreSQL, Oracle, SQL "
        "Server and SQLite; Python, Java, C++, JavaScript, C# and Go.",
        "FR7 — Answer free-text questions about the video, citing the "
        "moment each answer comes from.",
        "FR8 — Authenticate by e-mail one-time code, phone one-time code "
        "or Google, and enforce a per-tier allowance.",
        "FR9 — Resume a part-completed run from cached artefacts instead "
        "of recomputing completed stages.",
        "FR10 — Record every completed run as an observation and update "
        "the routed class's cost model from it.",
    ]:
        B(doc, r)

    SEC(doc, "3.1.2 Non-Functional Requirements", level=3)
    for r in [
        "NFR1 — The predicted finishing time must be produced before the "
        "job is enqueued, not after it starts.",
        "NFR2 — Updating a cost model must cost constant work per "
        "completed job, independent of how many jobs have been seen.",
        "NFR3 — A failure in any single stage must leave the artefacts "
        "of the stages that already succeeded, so the run can be resumed.",
        "NFR4 — A failed run must report the provider's actual error, "
        "not a generic message.",
        "NFR5 — Session tokens must be signed; one-time codes must be "
        "stored hashed and salted; payment callbacks must be verified by "
        "HMAC signature.",
        "NFR6 — The interface must be usable at a laptop width and must "
        "render correctly in both a light and a dark colourway.",
        "NFR7 — The system must remain correct under its automated test "
        "suite before any submission.",
    ]:
        B(doc, r)

    SEC(doc, "3.2 Feasibility Study")
    SEC(doc, "3.2.1 Technical Feasibility", level=3)
    J(doc,
      "Every component the project depends on is available and was exercised "
      "during development: speech recognition, scene detection, keyframe "
      "embedding, a vector index and an OpenAI-compatible language-model "
      "endpoint. The scheduler requires no special infrastructure — its "
      "state is a small table of running sums, and its decision is arithmetic "
      "over at most a handful of classes. The principal technical risk met "
      "during the project was not compute but provider behaviour: free "
      "language-model endpoints impose daily quotas, per-request size limits "
      "and reasoning-token accounting that silently consumes the output "
      "budget. Each of these was reproduced and handled explicitly rather "
      "than retried blindly.")

    SEC(doc, "3.2.2 Economic Feasibility", level=3)
    J(doc,
      "Development used free-tier language-model endpoints and a single "
      "laptop, so direct cost was negligible. The operating model assumed in "
      "the design is that compute is billed per hour and differs by class, "
      "which is what makes the cheapest-feasible-class objective meaningful. "
      "Section 7.5 gives the cost analysis. Payment integration was "
      "implemented and verified against the provider's live test API, so the "
      "revenue path is demonstrated rather than assumed.")

    SEC(doc, "3.2.3 Social Feasibility", level=3)
    J(doc,
      "The system addresses a real constraint on learners: time. By making "
      "long-form lecture content searchable, answerable and summarised, it "
      "serves Sustainable Development Goal 4, Quality Education, "
      "particularly for students whose access to live instruction is "
      "limited. Two risks were considered. The first is over-reliance: a "
      "generated explanation can be wrong, so every answer the system gives "
      "is cited back to the moment in the video it came from, allowing the "
      "claim to be checked. The second is language exclusion: pinning "
      "transcription to English was tried during development, degraded "
      "accuracy on code-mixed lectures and made them slower, and was "
      "replaced by automatic detection.")

    SEC(doc, "3.3 System Specification")
    SEC(doc, "3.3.1 Hardware Specification", level=3)
    TABLE(doc, "Hardware used for development and measurement", [
        ["Item", "Specification"],
        ["Development machine", "Windows 11, CPU-only, 16 GB RAM"],
        ["Worker classes modelled",
         "cpu-2core, cpu-8core, cpu-16core (emulated on one machine by "
         "constraining the execution plan)"],
        ["Accelerator", "None. GPU classes are supported by the routing "
                        "logic but were not available for measurement"],
        ["Storage", "Local disk for the stage artefact cache"],
    ], widths=[1.8, 4.3], font=10)
    J(doc,
      "The three worker classes were exercised on a single machine by "
      "constraining each class's execution plan to its own measured cost "
      "model. This is a faithful test of the routing logic and of the "
      "calibration loop, but it is not a test across physically distinct "
      "machines, and Chapter 7 treats it as such.")

    SEC(doc, "3.3.2 Software Specification", level=3)
    TABLE(doc, "Software stack", [
        ["Layer", "Technology", "Role"],
        ["Front end", "Next.js (App Router), React, TypeScript, Tailwind CSS",
         "The three-panel workspace, the entry screen and the studio"],
        ["API", "FastAPI, Pydantic, Uvicorn",
         "Job submission, authentication, quota checks, result serving"],
        ["Persistence", "SQLAlchemy over SQLite",
         "Users, jobs, payments, SLA observations, cost models"],
        ["Speech", "Whisper", "Transcription with language detection"],
        ["Vision", "Scene detection, keyframe extraction, YOLO, OCR",
         "What is on the screen, as distinct from what is said"],
        ["Retrieval", "FAISS, sentence embeddings",
         "Cited question answering over passages"],
        ["Understanding", "OpenAI-compatible LLM endpoint (OpenRouter)",
         "Explanation, quiz, flashcards, practice set"],
        ["Testing", "pytest", f"{F['tests']} automated tests"],
    ], widths=[1.1, 2.3, 2.7], font=9.5)

    # =============================================================== CHAPTER 4
    CH(doc, 4, "Design Approach and Details")
    SEC(doc, "4.1 System Architecture")
    J(doc,
      "The system is a pipeline of seven stages with a scheduler in front of "
      "it and a feedback path behind it. A submitted job is first priced: for "
      "each hardware class the scheduler predicts the total execution time "
      "from that class's own cost model, multiplies by the class's rate, and "
      "selects the cheapest class whose prediction fits inside the budget. "
      "The job is then executed, each stage writing its artefacts to a cache "
      "keyed by the content hash of the video. When the job finishes, the "
      "measured time is fed back into the routed class's cost model. That "
      "loop — predict, route, measure, refit — is the part of the "
      "design the project claims as its contribution.")
    FIGURE(doc, "figure1-architecture.png",
           "System architecture. The feedback path from completed jobs into "
           "the per-class cost models is drawn explicitly, because that loop "
           "is what makes the routing adaptive.")
    J(doc,
      "Stages M2 and M3 are concurrent: transcription and vision read the "
      "same media but share no state, so running them in sequence would "
      "waste the larger part of the wall-clock budget. The understanding "
      "stage M5′ is windowed — a long video is read in overlapping "
      "windows and the partial explanations are reduced — because a "
      "single prompt containing a three-hour transcript exceeds every "
      "provider's request limit.")
    FIGURE(doc, "figure3-routing.png",
           "The routing decision, as implemented. Exploration never overrides "
           "the deadline; among the classes that keep it, the cheapest is "
           "taken rather than the fastest.")
    J(doc,
      "The cost model is affine rather than proportional because a "
      "proportional model attributes the fixed cost of loading models to the "
      "length of the video, which makes short jobs look far more expensive "
      "per minute than they are and long jobs cheaper. The fit is maintained "
      "from five running sums, so a completed job costs constant work to "
      "absorb, and it is refused as not yet credible until at least four "
      "jobs have been seen whose lengths differ by a factor of about two "
      "— a line through four points that are all the same length "
      "carries no information about the slope.")
    FIGURE(doc, "figure2-cost-model.png",
           "The per-class execution-cost model, fitted from recorded runs. "
           "Runs resumed from cache are excluded, and the exclusion is "
           "printed on the figure.")

    SEC(doc, "4.2 Design")
    SEC(doc, "4.2.1 Data Flow Diagram", level=3)
    J(doc,
      "At the context level the system exchanges data with four external "
      "parties: the student, the source of the video, the language-model "
      "provider and the payment gateway.")
    FIGURE(doc, "fig-4-1-dfd-level0.png",
           "Context (level-0) data-flow diagram.")
    J(doc,
      "At level 1 the single process expands into seven, and four data "
      "stores appear. The store that matters to the contribution is D4, the "
      "per-class cost models: process 2 reads it to route and writes to it "
      "after every completed run.")
    FIGURE(doc, "fig-4-2-dfd-level1.png", "Level-1 data-flow diagram.")

    SEC(doc, "4.2.2 Use Case Diagram", level=3)
    J(doc,
      "Two human actors are distinguished by tier, and three system actors "
      "participate. The two shaded use cases are performed by the system "
      "itself rather than requested by a user; they are shown because they "
      "are where the project's contribution lies.")
    FIGURE(doc, "fig-4-3-use-case.png", "Use-case diagram.")

    SEC(doc, "4.2.3 Class Diagram", level=3)
    J(doc,
      "The persistent model is deliberately small. A Job records what was "
      "asked for, what was predicted, where it was routed and what it "
      "actually cost. An SlaObservation is the record from which calibration "
      "happens; SchedulerCost holds the running sums per class, and CostModel "
      "is the in-memory form that answers predict() and observe().")
    FIGURE(doc, "fig-4-4-class.png",
           "Class diagram of the persistent model and the scheduling classes.")

    SEC(doc, "4.2.4 Sequence Diagram", level=3)
    J(doc,
      "The order below is the one the code follows. The two highlighted "
      "messages are the read of the cost models before routing and the "
      "update of the affine fit after completion.")
    FIGURE(doc, "fig-4-5-sequence.png", "Sequence of one submitted job.")

    # =============================================================== CHAPTER 5
    CH(doc, 5, "Methodology and Testing")
    SEC(doc, "5.1 Module Description")
    TABLE(doc, "The seven pipeline stages and the scheduler", [
        ["Module", "Responsibility", "Principal output"],
        ["M1 Ingest", "Fetch the video or accept an upload, probe its "
                      "duration, extract the audio track.",
         "audio.wav, meta.json"],
        ["M2 Transcribe", "Speech recognition with automatic language "
                          "detection; long audio is chunked.",
         "transcript.json"],
        ["M3 Vision", "Scene detection, keyframe extraction, object "
                      "detection and on-screen text. Runs concurrently with "
                      "M2.", "scenes.json, keyframes/"],
        ["M4 Fuse", "Align each transcribed sentence to the scene it was "
                    "spoken over.", "fused.json"],
        ["M5′ Understand", "A language model reads the fused document in "
                                "windows and writes the explanation, quiz, "
                                "flashcards and practice set.",
         "understanding.json"],
        ["M6 Index", "Build a FAISS index over passages for cited question "
                     "answering.", "qa_index/"],
        ["M7 Highlight", "Score shots with the trained importance model and "
                         "cut a reel.", "highlight.mp4"],
        ["Scheduler", "Predict per-class cost, route the job, record the "
                      "outcome and refit.", "SlaObservation, SchedulerCost"],
    ], widths=[1.1, 3.1, 1.8], font=9.5)

    SEC(doc, "5.2 Testing")
    J(doc,
      f"The system is covered by {F['tests']} automated tests, all passing at "
      "the time of submission. The suite is weighted towards the parts of the "
      "system where a silent error would be most damaging: the cost model and "
      "the routing decision, where a wrong number produces a plausible but "
      "incorrect schedule rather than a crash.")
    TABLE(doc, "Test coverage by area", [
        ["Area", "What is tested"],
        ["Cost model", "Affine recovery from the running sums; refusal to use "
                       "a fit from too few samples or too narrow a spread of "
                       "lengths; rejection of a negative slope."],
        ["Routing", "Cheapest feasible class is chosen, not the fastest; "
                    "unmeasured classes are probed; exploration never "
                    "overrides the deadline; infeasibility is declared before "
                    "the run."],
        ["Calibration", "Runs resumed from cache do not update the cost "
                        "model; an implausible real-time ratio is rejected."],
        ["Pipeline", "Stage artefacts are cached and reused; a failed stage "
                     "leaves earlier artefacts intact; the real provider "
                     "error is carried into the failure message."],
        ["Understanding", "Window planning for long inputs; the fallback "
                          "overview when a reduce pass fails; empty-window "
                          "detection."],
        ["Identity and quota", "Code hashing and expiry; the resend window "
                               "runs from the last code sent; per-tier "
                               "allowances; payment signature verification."],
        ["API", "Schema validation, authentication on every route, quota "
                "refusals carrying a usable message."],
    ], widths=[1.4, 4.7], font=9.5)

    # =============================================================== CHAPTER 6
    CH(doc, 6, "Project Demonstration")
    J(doc,
      "The screens below are captures of the running application, taken at a "
      "fixed window size from a test account so that no personal data "
      "appears. The worked example throughout is a "
      f"{run['minutes']}-minute lecture, “{run['title']}”.")
    FIGURE(doc, "view-2-rear.png",
           "The entry screen, with the four headline figures measured from "
           "the system's own records.")
    FIGURE(doc, "view-7-submission.png",
           "The submission screen. The processing profile is chosen here, and "
           "the estimate is shown before the run starts.")
    FIGURE(doc, "view-1-front.png",
           "The workspace: source panel, the chaptered explanation, and the "
           "studio column of derived artefacts.")
    FIGURE(doc, "view-4-right.png",
           "The studio column. Each tile names a derived artefact and the "
           "material it was derived from, with the quantity on the right.",
           width=2.6)
    J(doc,
      "Figure captions for these screens correspond to the design "
      "representations submitted with the design-registration disclosure; the "
      "same captures are used in both documents so that the report and the "
      "filing cannot diverge.")

    # =============================================================== CHAPTER 7
    CH(doc, 7, "Result and Discussion")
    SEC(doc, "7.1 Throughput on a Worked Example")
    J(doc,
      f"The reference run is a {run['minutes']}-minute lecture processed end "
      f"to end on the {run['worker_class']} class. It completed in "
      f"{run['elapsed']} s against a budget of {int(run['sla'])} s, which is "
      f"about {run['speedup']}× faster than watching it.")
    tim = run["timings"]
    TABLE(doc, "Stage timings for the reference run, in seconds", [
        ["Stage", "Seconds", "Share of total"],
    ] + [[k, f"{v:.1f}", f"{v / tim['total'] * 100:.1f}%"]
         for k, v in tim.items() if k != "total"]
      + [["total", f"{tim['total']:.1f}", "100.0%"]],
      widths=[1.8, 1.4, 1.6], font=10)
    J(doc,
      "The distribution is the reason the scheduler exists. Analysis and "
      "understanding together account for the large majority of the time, "
      "and both scale with the length of the input; fusing and indexing are "
      "almost free. A cost model that captures the per-minute rate of the "
      "two dominant stages therefore predicts the whole run well enough to "
      "route on.")

    SEC(doc, "7.2 Deadline Attainment")
    J(doc,
      f"Over {s['observations']} recorded observations the scheduler met its "
      f"budget on {pct(s['met_rate'])} of jobs. The fixed-configuration "
      "baseline — a static, highest-fidelity plan that never downshifts "
      f"— met it on {pct(s['fixed_config_met'])}.")
    rows = [["Strategy", "SLA met", "Mean time ÷ budget", "Kind"],
            ["Scheduler (this project)", pct(s["met_rate"]),
             f"{s['mean_vs_sla']:.2f}", "Measured"],
            ["Fixed highest-fidelity configuration",
             pct(s["fixed_config_met"]), f"{s['fixed_config_vs_sla']:.2f}",
             "Measured"]]
    for cls_name, rate in sorted(s["always"].items()):
        rows.append([f"Always {cls_name}", pct(rate),
                     f"{s['always_vs_sla'][cls_name]:.2f}", "Counterfactual"])
    rows.append(["Random choice of class", pct(s["random"]), "—",
                 "Counterfactual"])
    TABLE(doc, "Deadline attainment against baselines", rows,
          widths=[2.4, 1.1, 1.5, 1.3], font=10)
    J(doc,
      "The last three rows must be read carefully, and the project does not "
      "claim more from them than they support. They are counterfactuals "
      "computed from each class's measured per-minute cost: for every "
      "recorded job they ask whether that class's predicted time would have "
      "fitted the budget. They therefore assume the cost model is exact. The "
      "scheduler's own figure is measured and includes every real-world "
      "failure — provider stalls, a reduce pass that had to be retried, "
      "a run whose summarise stage took nearly three hours.")
    J(doc,
      "Read that way, the honest conclusion is twofold. Against a static "
      "configuration that never adapts, the scheduler is decisively better: "
      f"{pct(s['met_rate'])} against {pct(s['fixed_config_met'])}. Against "
      "the strongest fixed routing choice — always using the largest "
      f"class — it is not better on attainment, and the counterfactual "
      f"suggests it would be worse ({pct(s['always'].get('cpu-16core', 0))}). "
      "The scheduler's claim is therefore comparable attainment at lower "
      "cost, which Section 7.5 quantifies, and not higher attainment. A "
      "direct measured A/B against the always-largest policy has not been "
      "run, and is the first item of future work.")

    SEC(doc, "7.3 A Limitation in the Budget Policy")
    J(doc,
      f"Every observation in the table above was taken against a flat budget "
      f"of {int(run['sla'])} s, applied regardless of the length of the "
      "video. For an eleven-minute lecture that is a demanding but reachable "
      "target. For a ten-hour tutorial it is unreachable by any "
      "configuration, and the system correctly declares so before starting. "
      "A substantial share of the recorded misses are therefore structural "
      "rather than failures of routing. A length-proportional budget would "
      "make the attainment figures more meaningful, and is listed in "
      "Chapter 8.")

    SEC(doc, "7.4 Summarisation Quality")
    J(doc,
      "Following Otani et al., F1 alone is not treated as evidence. The "
      "table reports F1 beside rank correlation, and beside two reference "
      "points computed on the same shards: the agreement between human "
      "annotators, and the score of random importance values.")
    tv, sm = ml["per_dataset"]["tvsum"], ml["per_dataset"]["summe"]
    bl = ml["baselines"]
    TABLE(doc, f"Summarisation quality, {ml['folds']}-fold cross-validation "
               f"over {ml['n_videos']} videos", [
        ["Dataset", "Model F1", "Human F1", "Random F1", "Model τ",
         "Model ρ"],
        ["TVSum", f"{tv['f1']:.3f}", f"{bl['tvsum']['human']:.3f}",
         f"{bl['tvsum']['random']:.3f}", f"{tv['kendall_tau']:.3f}",
         f"{tv['spearman_rho']:.3f}"],
        ["SumMe", f"{sm['f1']:.3f}", f"{bl['summe']['human']:.3f}",
         f"{bl['summe']['random']:.3f}", f"{sm['kendall_tau']:.3f}",
         f"{sm['spearman_rho']:.3f}"],
    ], widths=[1.1, 1.0, 1.0, 1.0, 1.0, 1.0], font=10)
    J(doc,
      "On TVSum the model's F1 is marginally above the human agreement "
      f"figure ({tv['f1']:.3f} against {bl['tvsum']['human']:.3f}) and well "
      f"above random ({bl['tvsum']['random']:.3f}), and its rank correlation "
      f"of τ = {tv['kendall_tau']:.3f} indicates it is genuinely "
      "ordering segments rather than exploiting the metric. On SumMe the "
      f"picture is different and worse: F1 of {sm['f1']:.3f} sits between "
      f"random ({bl['summe']['random']:.3f}) and human "
      f"({bl['summe']['human']:.3f}), but τ = {sm['kendall_tau']:.3f} is "
      "close to zero, meaning the model is not ranking SumMe shots "
      "meaningfully at all. This is exactly the failure mode Otani et al. "
      "warn about, and it is reported rather than hidden behind the "
      "favourable F1.")
    J(doc, ml["note"], )

    SEC(doc, "7.5 Cost Analysis")
    J(doc,
      "The scheduler optimises cost subject to the deadline, so the "
      "appropriate measure of its benefit is spend per job, not speed. Using "
      "the per-minute costs the system has measured for each class, the "
      "cheapest-feasible policy differs from the always-largest policy in "
      "that it places a job on a smaller class whenever the smaller class is "
      "also predicted to finish in time.")
    cm = {c["worker_class"]: c["seconds"] for c in s["cost_model"]
          if c.get("key") == "total:per_min"}
    if cm:
        TABLE(doc, "Measured per-minute execution cost by class", [
            ["Worker class", "Seconds per video-minute", "Samples"],
        ] + [[c["worker_class"], f"{c['seconds']:.2f}", int(c["samples"])]
             for c in s["cost_model"] if c.get("key") == "total:per_min"],
          widths=[2.0, 2.4, 1.2], font=10)
    J(doc,
      "Two caveats are owed here. The per-class figures were produced on one "
      "physical machine with the classes emulated, so they establish that the "
      "mechanism works and not what the savings would be on a real fleet. And "
      "the affine fit for the largest class is not yet credible by the "
      "system's own test — it has seen too few jobs at sufficiently "
      "different lengths — so routing currently falls back to the "
      "coarser per-minute ratio for that class. The fallback behaving "
      "correctly is itself one of the tested properties, but a fuller "
      "evaluation needs more jobs at a wider spread of lengths.")

    SEC(doc, "7.6 Prior Art and Patentability")
    J(doc,
      "Because the scheduler was considered for a patent filing, a prior-art "
      "search was carried out before any disclosure was submitted. It is "
      "reported here because its result qualifies the novelty claimed in "
      "Section 2.2.")
    J(doc,
      "The broad form of the claim — learn job duration from past runs, "
      "then choose the cheapest machine that still meets a deadline — "
      "appears anticipated by US 2014/0229221 A1, which was read in full. "
      "The joint selection of hardware and model fidelity is close to INFaaS. "
      "Linear regression for media-processing runtime is established in the "
      "transcoding literature. What no source was found to describe is the "
      "specific combination used here: an explicit fixed-overhead term with a "
      "per-input-minute rate, maintained per hardware class from running "
      "sums, gated by a credibility test on sample count and length spread, "
      "with asymmetric treatment of overruns and improvements, and "
      "exploration permitted only where the deadline is predicted to hold.")
    J(doc,
      "Invention disclosures for the scheduler and for the interface have "
      "been prepared on that basis, with the closest prior art cited in them "
      "rather than omitted. The formal classification search remains with the "
      "VIT IPR Cell.")

    # =============================================================== CHAPTER 8
    CH(doc, 8, "Conclusion")
    J(doc,
      "This project set out to make the time a video-understanding job takes "
      "a quantity the user is told in advance rather than discovers "
      "afterwards. It does that by learning, for each hardware class, an "
      "affine model of execution time against input length, fitting it online "
      "from completed jobs at constant work per job, and routing each new job "
      "to the cheapest class predicted to meet its budget — declaring "
      "the shortfall before the run starts when no class can.")
    J(doc,
      f"Measured over {s['observations']} runs, the scheduler meets its "
      f"budget on {pct(s['met_rate'])} of jobs against "
      f"{pct(s['fixed_config_met'])} for a static configuration. The "
      f"{run['minutes']}-minute reference lecture completes in "
      f"{run['elapsed']} s. The summarising model matches human-level F1 on "
      "TVSum with a rank correlation that confirms it is genuinely ordering "
      f"segments, and {F['tests']} automated tests pass.")
    J(doc,
      "Three limitations are stated plainly rather than deferred. The flat "
      "budget makes many of the recorded misses structural. The per-class "
      "comparison rests on counterfactuals computed from the cost model, not "
      "on a measured A/B, and against the strongest fixed policy the "
      "scheduler is not better on attainment. And the worker classes were "
      "emulated on a single machine, so the cost savings are demonstrated in "
      "mechanism rather than in magnitude.")
    J(doc, "The next steps follow directly from those limitations:")
    for f_ in [
        "A measured A/B against an always-largest-class policy, on physically "
        "distinct machines, replacing the counterfactual in Table 7.2.",
        "A length-proportional budget in place of the flat one, so that "
        "attainment measures routing rather than arithmetic.",
        "A second term in the cost model for output configuration — the "
        "transcoding literature shows execution time depends on target "
        "bitrate as well as duration.",
        "GPU worker classes, which the routing logic already supports and "
        "which would widen the spread of costs the scheduler has to reason "
        "about.",
        "Completion of the patent filing with the narrowed claim identified "
        "by the prior-art search.",
    ]:
        B(doc, f_)

    # =============================================================== CHAPTER 9
    CH(doc, 9, "References")
    refs = [
        "D. Potapov, M. Douze, Z. Harchaoui and C. Schmid, "
        "“Category-specific video summarization,” in Proc. European "
        "Conference on Computer Vision (ECCV), 2014, pp. 540–555.",
        "M. Otani, Y. Nakashima, E. Rahtu and J. Heikkilä, "
        "“Rethinking the evaluation of video summaries,” in Proc. "
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition "
        "(CVPR), 2019, pp. 7596–7604.",
        "A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey and "
        "I. Sutskever, “Robust speech recognition via large-scale weak "
        "supervision,” in Proc. International Conference on Machine "
        "Learning (ICML), 2023, pp. 28492–28518.",
        "J. Johnson, M. Douze and H. Jégou, “Billion-scale "
        "similarity search with GPUs,” IEEE Transactions on Big Data, "
        "vol. 7, no. 3, pp. 535–547, 2021.",
        "F. Romero, Q. Li, N. J. Yadwadkar and C. Kozyrakis, “INFaaS: "
        "Automated model-less inference serving,” in Proc. USENIX Annual "
        "Technical Conference (ATC), 2021, pp. 397–411.",
        "T. Deneke, H. Haile, S. Lafond and J. Lilius, “Video "
        "transcoding time prediction for proactive load balancing,” in "
        "Proc. IEEE International Conference on Multimedia and Expo (ICME), "
        "2014.",
        "“Cost-aware scheduling of compute jobs against a need-by "
        "time,” U.S. Patent Application US 2014/0229221 A1, Aug. 2014.",
        "“Estimating transcoding duration for scheduling media "
        "jobs,” U.S. Patent Application US 2015/0006226 A1, Jan. 2015.",
        "“System for content aware reconfiguration of video object "
        "detection,” U.S. Patent 12,469,288.",
        "Office of the Controller General of Patents, Designs and Trade "
        "Marks, Guidelines for Examination of Computer Related Inventions "
        "(CRIs). New Delhi, India: Government of India, 2025.",
        "Y. Song, J. Vallmitjana, A. Stent and A. Jaimes, “TVSum: "
        "Summarizing web videos using titles,” in Proc. IEEE Conference "
        "on Computer Vision and Pattern Recognition (CVPR), 2015, "
        "pp. 5179–5187.",
        "M. Gygli, H. Grabner, H. Riemenschneider and L. Van Gool, "
        "“Creating summaries from user videos,” in Proc. European "
        "Conference on Computer Vision (ECCV), 2014, pp. 505–520.",
    ]
    for i, r in enumerate(refs, 1):
        par = P(doc, space_after=6)
        par.paragraph_format.left_indent = m.Inches(0.4)
        par.paragraph_format.first_line_indent = m.Inches(-0.4)
        rr = par.add_run(f"[{i}]\t{r}")
        rr.font.name, rr.font.size = m.BODY_FONT, m.Pt(12)

    # ============================================================== APPENDIX A
    doc.add_page_break()
    P(doc, "APPENDIX A – SAMPLE CODE", size=16, bold=True, align=CENTER,
      space_after=14)
    J(doc,
      "Two extracts from server/scheduler.py: the recovery of the affine fit "
      "from the running sums together with the test that decides whether the "
      "fit may be used, and the routing decision itself.")
    SEC(doc, "A.1 Recovering the Affine Fit and Testing its Credibility")
    CODE(doc, FIT_CODE)
    SEC(doc, "A.2 The Routing Decision")
    CODE(doc, ROUTE_CODE)


FIT_CODE = '''n = stats["fit:n"][0]
sx, sy = stats["fit:sx"][0], stats["fit:sy"][0]
sxx, sxy = stats["fit:sxx"][0], stats["fit:sxy"][0]
if n < 2:
    return None

denom = n * sxx - sx * sx
if denom <= 1e-9:
    return None
slope = (n * sxy - sx * sy) / denom
intercept = (sy - slope * sx) / n

# x-spread, as max/min cannot be recovered from the sums: use the
# coefficient of variation as a proxy for "these were not all the same".
mean_x = sx / n
var_x = max(sxx / n - mean_x * mean_x, 0.0)
spread_ok = mean_x > 0 and (var_x ** 0.5) / mean_x > (_FIT_MIN_SPREAD - 1) / 3.0

credible = (
    n >= _FIT_MIN_SAMPLES
    and spread_ok
    and slope > 0
    and intercept >= -1e-6
)
return CostFit(max(intercept, 0.0), slope, int(n), credible)'''

ROUTE_CODE = '''# Explore, then exploit - but never at the SLA's expense, and among the
# classes that keep the promise, take the cheapest rather than the fastest.
#
# Choosing the fastest class that meets the SLA wastes the whole point of
# having a deadline: if a smaller, cheaper machine also finishes in time,
# the headroom bought by the larger one is paid for and never used.
#
# A class with no measurements has only a seed estimate, too crude to route
# on or to rule out, so probe it once. Beyond that, pure exploitation is a
# trap: the moment one class wins it takes every job, the others' models go
# stale, and the router is deciding on numbers it can no longer check.
unsampled = [p for p in predictions if p.samples == 0]
within = [p for p in predictions if p.meets_sla]
under_measured = [p for p in within if p.samples < _EXPLORE_MIN_SAMPLES]

by_cost = lambda p: (p.cost, p.predicted_sec, p.worker_class)
by_speed = lambda p: (p.predicted_sec, p.worker_class)

if unsampled:
    chosen = min(unsampled, key=by_cost)
elif under_measured:
    chosen = min(under_measured, key=lambda p: (p.samples, *by_cost(p)))
elif within:
    chosen = min(within, key=by_cost)
else:
    chosen = min(predictions, key=by_speed)'''
