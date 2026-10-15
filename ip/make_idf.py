"""Generate both disclosures strictly against the blank templates.

Headings, their numbering and their wording are copied verbatim from
IDF-B-template-blank.pdf and IDF-Design-template-blank.pdf, including the
table structures and the closing note. Only the answers are ours.

Run from the project root:  python ip/make_idf.py
"""

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

IP = "ip"

# ---------------------------------------------------------------- helpers


def new_doc():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(4)
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
    return doc


def title(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)


def heading(doc, text):
    """A numbered item, worded exactly as the blank form words it."""
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(3)


def body(doc, text, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    return p


def bullet(doc, text):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    return p


def shade(cell, hexcolor):
    el = cell._tc.get_or_add_tcPr()
    sh = el.makeelement(qn("w:shd"), {})
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), hexcolor)
    el.append(sh)


def figure(doc, path, caption):
    if not os.path.exists(path):
        body(doc, f"[{caption} — file not found at {path}]", italic=True)
        return
    doc.add_picture(path, width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)


# ================================================================ IDF-B
def build_idf_b():
    doc = new_doc()
    title(doc, "Invention Disclosure Form (IDF) -B – A Guide to Describing Your Innovation")

    # ---- 1
    heading(doc, "1. Title of the invention:")
    body(doc,
         "Deadline-constrained scheduling of multimodal video-understanding workloads by "
         "jointly selecting a hardware class and a model-fidelity configuration, using a "
         "per-class execution-cost model learned online as an affine function of input "
         "length.")
    body(doc,
         "Short title: Cost-aware, SLA-constrained scheduler with online affine cost learning "
         "for heterogeneous inference fleets.")

    # ---- 2
    heading(doc, "2. Field / Area of invention:")
    body(doc,
         "Distributed computing and resource allocation; scheduling of machine-learning "
         "inference workloads across heterogeneous hardware; adaptive quality-of-service "
         "systems. Reduced to practice in an automated video-understanding pipeline combining "
         "automatic speech recognition, computer vision and large-language-model "
         "summarisation.")

    # ---- 3
    heading(doc, "3. Prior Patents and Publications (Table summarizing prior art):")
    body(doc,
         "No patent numbers are cited below, because none have been independently verified by "
         "the inventors. A formal prior-art search (Indian and international) has not been "
         "carried out and must be completed through the VIT IPR Cell before filing. The "
         "entries are the academic work actually relied upon during development, and the "
         "search categories the inventors believe the examiner should cover.", italic=True)

    rows = [
        ("Patent/Publication", "Description", "Relevance to your invention"),
        ("Apostolidis et al., Proceedings of the IEEE, 109(11), 2021",
         "Survey of deep-learning video summarisation.",
         "States the art in selecting salient video content. Concerned with what to select, "
         "not with the wall-clock cost of producing it on given hardware."),
        ("Otani et al., CVPR 2019",
         "Re-evaluation of video-summarisation metrics; rank correlation against each "
         "annotator.",
         "The evaluation protocol adopted for the scene-importance component reported in "
         "item 8."),
        ("Potapov et al., ECCV 2014",
         "Kernel Temporal Segmentation (KTS) for shot boundaries.",
         "Used unmodified as a segmentation primitive. Not claimed."),
        ("Radford et al., 2022 (Whisper); Redmon et al., 2016 (YOLO); "
         "Radford et al., 2021 (CLIP)",
         "Speech recognition, object detection and image-text embedding models.",
         "Used as interchangeable components. The invention selects among such models under a "
         "time budget; it does not claim the models."),
        ("Prior-art categories requiring search by the IPR Cell "
         "(cite related Indian and other countries patents)",
         "(a) SLA-aware or deadline-aware job scheduling in cloud and cluster systems; "
         "(b) online regression of job runtime from input features; (c) accuracy-versus-"
         "latency model selection, including anytime inference and model cascades; "
         "(d) serverless cold-start and execution-time prediction.",
         "The fields in which the closest art is expected. The inventors' position is that the "
         "combination in item 9 — an affine per-class cost model fitted online from media "
         "duration, driving joint hardware-class and model-fidelity selection against a "
         "wall-clock deadline — is what must be searched, not scheduling in general."),
    ]
    t = doc.add_table(rows=0, cols=3)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (a, b, c) in enumerate(rows):
        cells = t.add_row().cells
        for cell, text in zip(cells, (a, b, c)):
            cell.text = text
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                for r in p.runs:
                    r.font.size = Pt(9)
                    if i == 0:
                        r.bold = True
            if i == 0:
                shade(cell, "E7E9EC")

    # ---- 4
    heading(doc, "4. Summary and background of the invention (Address the gap/Novelty):")
    body(doc,
         "Problem. Understanding a long recorded video automatically requires several "
         "expensive stages — speech recognition, scene analysis, and summarisation by a large "
         "language model. On a fixed machine the total time grows with the length of the "
         "video, so a system that is responsive on a ten-minute clip becomes unusable on a "
         "ten-hour lecture. Where several machines of differing capability are available, "
         "which machine runs a given job, and at what model fidelity, determines whether the "
         "job finishes within an acceptable time at all.")
    body(doc,
         "Existing solutions and their limitations. Cluster schedulers route work by queue "
         "length, resource request or fair share, and treat execution time as an attribute of "
         "the job supplied by the user. Where runtime is predicted, it is commonly modelled "
         "as a constant rate per unit of input — so many seconds per minute of media. That "
         "assumption fails in this workload because a run pays a substantial fixed cost "
         "before it processes anything: loading models into memory, starting worker "
         "processes, and one round trip to a remote language model. Dividing a measured total "
         "by the input length folds that fixed cost into the rate, and the resulting rate "
         "depends on how long the observed videos happened to be.")
    body(doc,
         "Gap and novelty. The invention models the execution cost of each hardware class as "
         "an affine function of input length, total = fixed + rate × minutes, fitted online "
         "by ordinary least squares and updated in constant time per completed job. It then "
         "selects jointly — in one decision — both the hardware class and the fidelity of the "
         "models to run, choosing the least costly combination predicted to meet a wall-clock "
         "deadline. A second mechanism keeps the prediction honest on the machine it is "
         "running on, by feeding each stage's measured cost back into the planner with a "
         "deliberately asymmetric update in which evidence of slowness is believed faster "
         "than evidence of speed. The effect of the two together is measured in item 8.")

    # ---- 5
    heading(doc, "5. Objective(s) of Invention:")
    body(doc,
         "1. To complete an automated video-understanding job within a stated wall-clock "
         "deadline on hardware that is heterogeneous, shared and not known in advance.")
    body(doc,
         "2. To predict the execution cost of a job on each available hardware class "
         "accurately enough for that choice to be made before the job starts, and from "
         "observations of jobs whose input lengths differ widely.")
    body(doc,
         "3. To spend the least costly resource that still meets the deadline, rather than "
         "the fastest resource available.")
    body(doc,
         "4. To require no manual calibration when the system is moved to different hardware, "
         "by learning the machine's behaviour from its own completed work.")
    body(doc,
         "5. To degrade predictably rather than fail when the deadline cannot be met by any "
         "configuration, by reporting the shortfall in advance.")

    # ---- 6
    heading(doc, "6. Working principle of the invention (in brief):")
    body(doc,
         "When a job arrives, the system reads the length of the input media from its "
         "metadata, without downloading it. For every hardware class in the fleet it predicts "
         "the total execution time from that class's learned affine cost model, and "
         "multiplies the prediction by the class's hourly rate to obtain a cost. Classes "
         "whose prediction exceeds the deadline are set aside. Among those that remain, the "
         "cheapest is chosen; a class with no measurements yet is sampled in preference to "
         "re-using one already characterised, so the fleet learns about itself. If no class "
         "meets the deadline, the fastest is chosen and the expected overrun is reported "
         "rather than concealed.")
    body(doc,
         "On the chosen machine, a second decision selects how heavy the models may be — the "
         "speech-recognition checkpoint and its beam width, the degree of parallel "
         "transcription, the number of keyframes analysed, the summarisation model — so that "
         "the predicted total fits the remaining budget. When the job completes, each stage's "
         "measured cost is written back: the five running sums of that class's cost model are "
         "updated, and the planner's per-stage estimates are adjusted.")

    # ---- 7
    heading(doc,
            "7. Description of the invention in detail (Include drawings and/or photographs "
            "as needed):")

    body(doc, "7.1 The learned cost model.", italic=True)
    body(doc,
         "Each hardware class carries five numbers: the count of observations and the running "
         "sums Σx, Σy, Σxx and Σxy, where x is the input length in minutes and y is the "
         "measured total execution time in seconds. The affine fit is recovered from these by "
         "ordinary least squares, so a completed job updates the model in constant time and "
         "no history need be retained. Two guards decide whether the fit may be used: at "
         "least four observations, and a spread in observed lengths of at least two to one. A "
         "straight line through points of nearly equal length has a slope determined by "
         "noise, and the system falls back to a seeded estimate rather than trusting it.")

    body(doc, "7.2 Routing.", italic=True)
    body(doc,
         "A routing decision records, for each class, the predicted time, whether it meets "
         "the deadline, the resulting cost, and whether the prediction came from measurement "
         "or from a seed. The class finally chosen, its prediction, the deadline and the "
         "eventual measured time are all persisted, which is what makes the comparison in "
         "item 8 possible.")

    body(doc, "7.3 The self-calibrating planner.", italic=True)
    body(doc,
         "The planner holds a per-stage estimate of cost per minute of media for the machine "
         "it runs on, seeded from a hardware probe and corrected by every completed run. The "
         "correction is asymmetric: an observation slower than the current estimate moves it "
         "quickly, an observation faster moves it slowly. A symmetric update leaves the "
         "estimate systematically optimistic, because the fast observations are the ones that "
         "occur when the machine is otherwise idle, and an optimistic budget produces "
         "overruns.")

    body(doc, "7.4 Secondary embodiment — metered external model services.", italic=True)
    body(doc,
         "Where a stage is served by a metered external language model rather than by local "
         "computation, the same deadline discipline is applied to a different scarce "
         "resource: the number of requests permitted per period. A long video is divided into "
         "windows whose count is capped by the request budget rather than by the content, so "
         "the cost of one job is bounded. The system distinguishes a per-period exhaustion of "
         "quota, which will not clear within the run, from a momentary rate limit, which "
         "will; it preserves work already completed when quota is exhausted part-way; and it "
         "continues to a further configured account rather than failing. The technical effect "
         "is the completion of jobs that would otherwise abort, and the elimination of "
         "waiting on a limit that cannot clear.")

    body(doc, "Drawings.", italic=True)
    figure(doc, f"{IP}/figure1-architecture.png",
           "Figure 1 — System architecture. The highlighted loop is the invention: every "
           "completed job updates its hardware class's cost model, and that model decides "
           "where the next job runs.")
    figure(doc, f"{IP}/figure2-cost-model.png",
           "Figure 2 — Measured execution cost against input length, per hardware class, with "
           "the fitted affine model and a constant-rate model fitted only on short inputs.")
    figure(doc, f"{IP}/figure3-routing.png",
           "Figure 3 — The routing decision as implemented: the three sources a prediction "
           "may come from, the two exploration branches, the cheapest-that-meets-it rule and "
           "the deadline-miss path.")

    # ---- 8
    heading(doc, "8. Experimental validation results:")
    body(doc,
         "All figures below are read from the system's own recorded state and are "
         "reproducible from the repository.", italic=True)

    body(doc, "8.1 Deadline attainment.", italic=True)
    body(doc,
         "Thirty-one jobs are recorded, each storing its prediction, its deadline, its "
         "measured time and the time a fixed baseline configuration would have taken. Six of "
         "those resumed from cached work and are excluded throughout, because a run that did "
         "only part of the work says nothing about throughput; the exclusion is stated here "
         "rather than left silent. Over the remaining 25 full runs the adaptive scheduler met "
         "its deadline on 68.0% of jobs against 8.0% for the fixed baseline configuration on "
         "identical hardware. Single-class routing policies recorded over the same jobs met "
         "the deadline on 60.0% (always the fastest class), 53.3% (middle class) and 10.0% "
         "(slowest class), showing that the gain comes from the choice of class and not "
         "merely from a fast machine being available.")

    body(doc, "8.2 Accuracy of the cost model.", italic=True)
    body(doc,
         "Figure 2 plots every full run against input length, per hardware class. On the "
         "class with the most observations — 16 runs, over inputs from 1.6 to 10.8 minutes — "
         "a constant-rate model fitted only on the videos of five minutes or less yields 25.5 "
         "seconds per video-minute. The affine model fitted over all 16 yields "
         "total = 48 s + 9.3 s × minutes. At 10.8 minutes, the longest input this class "
         "actually ran, the measured time was 141 seconds: the affine model predicts 148 "
         "seconds, an error of 5%, and the constant-rate model predicts 275 seconds, an error "
         "of 95%. The divergence is therefore demonstrated within the range of measured data "
         "and does not rest on extrapolation.")
    body(doc,
         "The cause is visible in the fitted intercept of 48 seconds, which is the cost a run "
         "pays before it processes anything. Dividing a measured total by input length folds "
         "that cost into the rate, so a rate inferred from short videos carries it, and the "
         "resulting error grows with the length of the video being predicted.")

    body(doc, "8.3 End-to-end throughput.", italic=True)
    body(doc,
         "A full run with no reuse of cached work, measured through the deployed service: an "
         "11.4-minute lecture completed in 159.2 seconds, that is 4.3 times faster than "
         "watching it, against a 300-second deadline which was met. Stage costs were ingest "
         "16.7 s, combined transcription and visual analysis 69.8 s, alignment 0.1 s, "
         "summarisation 43.5 s, indexing 2.9 s. A 10-hour course processed in 2 hours 10 "
         "minutes, 4.6 times faster than watching, so the ratio holds across two orders of "
         "magnitude of input length.")

    body(doc, "8.4 Component validation — scene-importance model.", italic=True)
    body(doc,
         "The learned scene-importance component, which orders the visual material passed to "
         "the summarisation stage, was evaluated by five-fold cross-validation on the SumMe "
         "and TVSum datasets using the rank-correlation protocol of Otani et al. (2019). Mean "
         "Kendall τ = 0.160 ± 0.016; summary F1 = 0.323 ± 0.034. Measured against baselines "
         "computed inside the same pipeline: on TVSum the model attains F1 0.265 where random "
         "scoring attains 0.166 and human inter-annotator agreement attains 0.260, and its "
         "per-annotator τ of 0.216 exceeds the approximately 0.18 that annotators achieve "
         "against one another; on SumMe the model attains 0.451 against 0.368 for random and "
         "0.612 for human. Absolute F1 is not comparable across summarisation pipelines, "
         "which is why the baselines were computed within this one.")

    body(doc, "8.5 Software verification.", italic=True)
    body(doc,
         "The implementation is covered by 190 automated tests, which pass in full. These "
         "include regression tests for each failure mode described in item 7.4 and for the "
         "credibility guards on the cost fit described in item 7.1.")

    # ---- 9
    heading(doc, "9. What aspect(s) of the invention need(s) protection?")
    body(doc,
         "The inventors are advised that under Section 3(k) of the Patents Act, 1970 a "
         "computer programme per se, an algorithm or a mathematical method is not patentable, "
         "and that protection is therefore sought for the technical effect on resource "
         "utilisation of a computing system, not for the computation itself. The aspects "
         "below are stated in that form and in order of importance.", italic=True)

    body(doc, "Principal aspect.", italic=True)
    body(doc,
         "A method of scheduling media-processing jobs across a plurality of hardware classes "
         "in which, for each class, the execution cost is maintained as an affine function of "
         "input media duration, fitted online by least squares from running sums updated in "
         "constant time per completed job; and in which a hardware class and a model-fidelity "
         "configuration are selected jointly, before the job begins, as the least costly "
         "combination whose predicted execution time satisfies a wall-clock deadline.")

    body(doc, "Dependent aspects.", italic=True)
    bullet(doc,
           "The credibility conditions under which the fitted model is used rather than a "
           "seeded estimate — a minimum observation count and a minimum spread of observed "
           "input lengths — and the fallback behaviour when they are not satisfied.")
    bullet(doc,
           "The preference for sampling a hardware class with no measurements over re-using a "
           "characterised one, so that the fleet becomes self-describing without a "
           "calibration run.")
    bullet(doc,
           "The asymmetric updating of the per-stage cost estimates, in which evidence of "
           "slower execution is incorporated more rapidly than evidence of faster execution, "
           "and its effect of preventing systematic optimism in the budget.")
    bullet(doc,
           "The exclusion from calibration of a run that reused cached results, so that a "
           "partial run cannot make a hardware class appear faster than it is.")
    bullet(doc,
           "The reporting of an expected deadline overrun in advance where no configuration "
           "satisfies the deadline, in place of accepting the job and failing to meet it.")

    body(doc, "Secondary aspect (may be filed as a separate embodiment).", italic=True)
    bullet(doc,
           "The application of the same deadline discipline to a metered external inference "
           "service, comprising: bounding the number of requests a single job may consume by "
           "dividing the input into a capped number of windows; distinguishing per-period "
           "quota exhaustion from a momentary rate limit and ceasing to wait on the former; "
           "preserving the work completed before exhaustion; and continuing on a further "
           "configured account rather than terminating the job.")

    body(doc, "Not claimed.", italic=True)
    body(doc,
         "The speech-recognition, object-detection, image-embedding and language models are "
         "used as interchangeable components and are not claimed. The segmentation primitive "
         "is prior art and is not claimed. The user interface is the subject of a separate "
         "design registration disclosure.")

    # ---- 10
    heading(doc,
            "10. What is Technology readiness level of your invention? (Tick the appropriate "
            "TRL)")

    trl = doc.add_table(rows=4, cols=9)
    trl.style = "Table Grid"
    trl.alignment = WD_TABLE_ALIGNMENT.CENTER

    groups = [("Research", 0, 2), ("Development", 3, 5), ("Deployment", 6, 8)]
    for name, a, b in groups:
        merged = trl.rows[0].cells[a].merge(trl.rows[0].cells[b])
        merged.text = name
        p = merged.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].bold = True
        shade(merged, "E7E9EC")

    for i in range(9):
        c = trl.rows[1].cells[i]
        c.text = f"TRL {i + 1}"
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraphs[0].runs[0].bold = True
        c.paragraphs[0].runs[0].font.size = Pt(8.5)

    desc = [
        "Basic Principles observed",
        "Technology concept formulated",
        "Experimental proof of concept",
        "Technology validated in a lab",
        "Technology validated in a relevant environment (industrially relevant in case of key "
        "enabling technologies)",
        "Technology demonstrated in a relevant environment (industrially relevant in case of "
        "key enabling technologies)",
        "System prototype demonstration in an operational environment",
        "System complete and qualified",
        "Actual system proven in an operational environment (competitive manufacturing in "
        "case of key enabling technologies, or in space)",
    ]
    for i, d in enumerate(desc):
        c = trl.rows[2].cells[i]
        c.text = d
        for r in c.paragraphs[0].runs:
            r.font.size = Pt(7)

    for i in range(9):
        c = trl.rows[3].cells[i]
        c.text = "✓" if i == 6 else ""
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if i == 6:
            c.paragraphs[0].runs[0].bold = True
            c.paragraphs[0].runs[0].font.size = Pt(14)
            shade(c, "FFF2CC")

    body(doc, "")
    body(doc,
         "TRL 7 — system prototype demonstration in an operational environment. The complete "
         "pipeline, the scheduler and the account-and-quota layer are deployed and reachable "
         "over the public internet, have processed videos up to 11 hours in length, and have "
         "served real sign-ins and verified payment transactions in a payment gateway's test "
         "mode. It is not yet TRL 8: the deployment runs from a single machine through a "
         "tunnelled endpoint, has not been qualified for sustained multi-user load, and its "
         "highlight-generation component depends on a model checkpoint not distributed with "
         "the source.")

    body(doc, "")
    body(doc, "Disclosure status relevant to filing (for the attention of the IPR Cell):",
         italic=True)
    body(doc,
         "The source repository is private. However, the running system has been made "
         "reachable over the public internet through a tunnelled endpoint for demonstration "
         "and testing, without access control on its landing page. The inventors record this "
         "as a matter of fact and seek the Cell's determination as to whether it constitutes "
         "a publication or public disclosure affecting novelty, and whether the endpoint "
         "should be withdrawn or placed behind access control pending filing. No paper, "
         "poster, thesis or public presentation describing the invention has been published "
         "to date.")

    body(doc, "")
    p = doc.add_paragraph()
    r = p.add_run(
        "Note: This is just a template, and the specific details you include will vary "
        "depending on your invention. Be sure to provide complete and accurate information "
        "to effectively describe your invention.")
    r.italic = True
    r.font.size = Pt(9)

    out = f"{IP}/IDF-B-Patent-SLA-Scheduler-FILLED.docx"
    doc.save(out)
    print(f"wrote {out}")


# ================================================================ DESIGN
def build_design():
    doc = new_doc()
    title(doc, "Invention Disclosure Format-Design Registration")

    meta = doc.add_table(rows=0, cols=2)
    meta.style = "Table Grid"
    for k, v in (("Document No.", "04-IPR-R001"),
                 ("Issue No/Date", "2/20.12.2024"),
                 ("Amd. No/Date", "0/00.00.0000")):
        cells = meta.add_row().cells
        cells[0].text = k
        cells[1].text = v
        cells[0].paragraphs[0].runs[0].bold = True

    # ---- 1
    heading(doc, "1. Please suggest a descriptive title for your design:")
    body(doc,
         "Graphical user interface for a video-understanding workspace, having a three-panel "
         "arrangement comprising a source panel, a central canvas and a studio panel of "
         "generated-artefact tiles, together with an associated device mark.")

    # ---- 2
    heading(doc,
            "2. A brief write-up on the article (For understanding of nature and novelty of "
            "the article).")
    body(doc,
         "The article is the visual layout and ornamentation of the screen display of a "
         "video-understanding application. Registration is sought for the appearance of the "
         "display — its arrangement, proportion and ornament — and not for the underlying "
         "function.")
    body(doc,
         "The working screen divides into three vertical panels that scroll independently of "
         "one another. At the left, a source panel carries the video player, a row of "
         "statistics beneath it and a chapter list. At the centre, a canvas carries tabbed "
         "views of the generated explanation, a scene grid, the transcript, a conversation "
         "and the reader's notes. At the right, a studio panel presents a vertical column of "
         "uniform tiles, each pairing a coloured round-cornered icon chip with a two-line "
         "label that names a derived artefact on the first line and, in smaller dimmer type, "
         "the material it was derived from on the second.")
    body(doc, "Features considered distinctive:", italic=True)
    for f in [
        "The three-panel division into independently scrolling regions, each panel rendered "
        "as a round-cornered card raised from a common ground rather than as a flush column.",
        "The studio tile column: tiles of equal height, each with a coloured round-cornered "
        "icon chip at the left, a two-line label, and a right-aligned numeral in monospaced "
        "figures giving the quantity of material behind that artefact.",
        "The two-line tile label, in which the second line names the source material in "
        "smaller dimmer type, applied uniformly across every tile in the column.",
        "An entry screen whose hero is centred beneath the device mark and is immediately "
        "followed by a full-width band of four figures, each figure set in monospaced "
        "numerals above a bold caption and a smaller explanatory line, the four divided by "
        "hairline rules into equal cells.",
        "A notice panel style used for a decision the reader must make — a round-cornered "
        "block in a warning tint, headed by a small icon and bold line, carrying a row of "
        "pill labels in which completed items are ticked, and closing with two or three "
        "buttons.",
        "An account panel that drops from the right of the header as a round-cornered card, "
        "presenting plan, remaining allowance and limit as label-and-value rows above a "
        "full-width action button.",
        "The device mark: a round-cornered square bearing three horizontal bars of "
        "diminishing width, their right ends falling along the diagonal of a play triangle, "
        "finished in a diagonal gradient.",
    ]:
        bullet(doc, f)
    body(doc,
         "The same arrangement, proportions and ornament are presented in a light and a dark "
         "colourway. Registration is sought for the arrangement and ornament; the colourways "
         "are submitted as variants of one design.")

    # ---- 3
    heading(doc,
            "3. Please share Soft copies of color photographs (with white background) "
            "depicting the design exactly through the following views:")
    for v in ["(i) a front view,", "(ii) a rear view,", "(iii) a left side view,",
              "(iv) a right side view,", "(v) a top view,", "(vi) a bottom view and",
              "(vii) a top perspective view (take any top left or top right-angle view at 45 "
              "degree from top position)."]:
        body(doc, v)
    body(doc, "")
    body(doc,
         "The subject of this design is a screen display, not a solid article, so the six "
         "orthographic views of an object do not apply. The corresponding screen "
         "representations are listed below; the IPR Cell's guidance is requested on the view "
         "set and numbering it prefers before filing.", italic=True)

    vt = doc.add_table(rows=0, cols=3)
    vt.style = "Table Grid"
    views = [
        ("View", "Representation", "Status"),
        ("(i)", "Front view — the working three-panel screen with a processed video open, "
                "light theme", "To be captured"),
        ("(ii)", "Rear view — not applicable; the entry screen is submitted in its place, "
                 "being the opposite face of the application as a user encounters it",
         "To be captured"),
        ("(iii)", "Left side view — the source panel at full height, shown alone",
         "To be captured"),
        ("(iv)", "Right side view — the studio tile column at full height, shown alone",
         "To be captured"),
        ("(v)", "Top view — the header bar with the device mark, navigation and account "
                "control", "To be captured"),
        ("(vi)", "Bottom view — the entry screen's figure band and footer", "To be captured"),
        ("(vii)", "Top perspective view — the working screen with the account panel open "
                  "over it", "To be captured"),
        ("—", "Device mark, supplied separately at not less than 1000 × 1000 pixels",
         "To be captured"),
    ]
    for i, (a, b, c) in enumerate(views):
        cells = vt.add_row().cells
        for cell, text in zip(cells, (a, b, c)):
            cell.text = text
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                for r in p.runs:
                    r.font.size = Pt(9)
                    if i == 0:
                        r.bold = True
            if i == 0:
                shade(cell, "E7E9EC")
    body(doc, "")
    body(doc,
         "To be captured at a single fixed window size, on a white background, with no "
         "browser chrome, no cursor and no personal data visible.", italic=True)

    # ---- 4
    heading(doc,
            "4. Has the design been published or disclosed to anyone outside of your "
            "organization?")
    body(doc,
         "The inventors record the following as matters of fact and seek the IPR Cell's "
         "determination, the question being a legal one:", italic=True)
    bullet(doc, "The source repository is private and has not been made public.")
    bullet(doc,
           "The running application has been made reachable over the public internet through "
           "a tunnelled endpoint for demonstration and testing. Its entry screen, which "
           "embodies part of the design, is served without any access control. The endpoint "
           "address was not published, listed or submitted to any search engine, and access "
           "was limited in practice to the inventors and their guide.")
    bullet(doc,
           "No paper, poster, thesis, presentation or exhibition describing or depicting the "
           "design has been published to date.")
    body(doc,
         "The inventors request the Cell's advice on whether the tunnelled endpoint "
         "constitutes prior publication for the purposes of Section 4 of the Designs Act, "
         "2000, and whether it should be withdrawn or placed behind access control pending "
         "the application.", italic=True)

    # ---- 5
    heading(doc,
            "5. Is there any such publication or disclosure planned? If so, provide date and "
            "details.")
    body(doc,
         "The design is expected to be shown in the Project-I (BCSE497J) review presentation "
         "and in the associated project report submitted to the School of Computer Science "
         "and Engineering, VIT Vellore. No other publication, exhibition or public release is "
         "planned.")
    body(doc, "Date of the review presentation: ______________________", italic=True)

    # ---- 6
    heading(doc, "6. What products or processes currently implement your design?")
    body(doc,
         "The design is implemented in the screen display of the AI-Based Video Summarization "
         "System developed for Project-I (BCSE497J). The application comprises a web front "
         "end and a service tier, and the design is embodied in the front end as rendered in "
         "a browser. It is implemented in no other product, and has not been licensed or "
         "supplied to any third party.")

    # ---- 7
    heading(doc,
            "7. Please provide details of each person(s) who has contributed to the "
            "conception of the design:")
    for n in range(1, 4):
        if n > 1:
            body(doc, "")
        for field, value in (("Name :", ""),
                             ("Title   :", "Student, B.Tech"),
                             ("Organization:", "Vellore Institute of Technology, Vellore"),
                             ("Address:", ""),
                             ("Email:", ""),
                             ("Phone:", "")):
            body(doc, f"{field} {value}" if value else f"{field} ______________________________")

    # ---- 8
    heading(doc, "8. Signature of inventor(s) and date")
    for _ in range(3):
        body(doc, "")
        body(doc, "-------------------------------------------------------------------")

    out = f"{IP}/IDF-Design-Registration-Interface-FILLED.docx"
    doc.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    build_idf_b()
    build_design()
