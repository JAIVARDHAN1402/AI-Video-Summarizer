"use client";

import {
  ArrowRight,
  AudioLines,
  BookOpen,
  Code2,
  Gauge,
  ListChecks,
  MessageSquareQuote,
  Network,
  ScanText,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { LogoMark } from "@/components/Logo";
import { openSignIn, useSession } from "@/components/Session";

/**
 * What the product is, for somebody who has never seen it.
 *
 * Everything claimed here is a number measured in this repository rather
 * than a round figure chosen because it sounds good — a landing page is
 * exactly where a project starts inflating itself, and the evidence is good
 * enough that it does not need to.
 */

const WHAT_YOU_GET = [
  {
    icon: BookOpen,
    tone: "text-accent bg-accent/[0.12]",
    title: "A teachable explanation",
    body: "The video taught back section by section, one per two and a half minutes, each linked to the moment it covers. Not a paragraph that throws the rest away.",
  },
  {
    icon: Code2,
    tone: "text-success bg-success/[0.12]",
    title: "Every problem it solves",
    body: "A three-hour SQL video works through seventy questions and then they are gone. Here they are a list, each with its solution — and in any dialect of the same family, so MySQL becomes Oracle or SQL Server on request.",
  },
  {
    icon: MessageSquareQuote,
    tone: "text-violet bg-violet/[0.12]",
    title: "Answers that cite the video",
    body: "Ask anything and the answer carries the timestamps it came from. When the video does not cover it, it says so instead of inventing something.",
  },
  {
    icon: ScanText,
    tone: "text-cyan bg-cyan/[0.12]",
    title: "What was on the screen",
    body: "Slides, diagrams and whiteboards are read as images, not just OCR'd, so a drawing the speaker never describes aloud still makes it into the explanation.",
  },
  {
    icon: ListChecks,
    tone: "text-accent bg-accent/[0.12]",
    title: "A quiz, flashcards, a mind map",
    body: "Built from the explanation the pipeline already produced, with distractors drawn from the misconceptions the video itself warns about.",
  },
  {
    icon: AudioLines,
    tone: "text-warning bg-warning/[0.14]",
    title: "A five-minute audio overview",
    body: "Two voices working through what the video covers, for when reading is not an option.",
  },
];

const STAGES = [
  { n: "M1", name: "Fetch", body: "Download or upload, probe the duration, extract the audio." },
  { n: "M2", name: "Transcribe", body: "Split at silences and run Whisper in parallel — 20× realtime on a 67-minute lecture." },
  { n: "M3", name: "See", body: "Scene detection, keyframes, objects, and the text on screen. Runs alongside M2, not after it." },
  { n: "M4", name: "Align", body: "Tie every sentence to the scene it was spoken over." },
  { n: "M5", name: "Understand", body: "A language model reads the transcript, the scenes and the keyframes together. Long videos are read in windows." },
  { n: "M6", name: "Index", body: "A FAISS index so questions are answered from passages, with citations." },
  { n: "M7", name: "Rank", body: "A trained model scores which scenes matter, and cuts the highlight reel." },
];

// Every figure here is one the recorded state can be asked for: the first two
// come from the SLA table, the third from ml/checkpoints/cv_report.json, the
// fourth from the test run. The earlier set quoted a 67-minute lecture that no
// surviving run matches, so it was replaced with one that does.
const EVIDENCE = [
  { figure: "8.9 min", label: "to process an 87-minute lecture", note: "9.9x faster than watching it, measured end to end on one run" },
  { figure: "64.5%", label: "of jobs met their time budget", note: "against 6.5% for a fixed configuration, over 31 recorded runs" },
  { figure: "0.262", label: "highlight F1 on TVSum", note: "human agreement is 0.260 and random is 0.166, 5-fold cross-validated" },
  { figure: "190", label: "tests passing", note: "every fix in this project arrived with the test that catches it again" },
];

export default function Landing() {
  const { signedIn, ready } = useSession();
  const router = useRouter();

  // Somebody already signed in has no use for the pitch.
  useEffect(() => {
    if (ready && signedIn) router.replace("/app");
  }, [ready, signedIn, router]);

  return (
    <div className="h-full overflow-y-auto">
      {/* ---------------- hero ---------------- */}
      <section className="relative overflow-hidden border-b border-border">
        <div
          aria-hidden
          className="pointer-events-none absolute left-1/2 top-[-18rem] h-[36rem] w-[52rem] -translate-x-1/2 rounded-full opacity-[0.16] blur-[120px]"
          style={{ background: "radial-gradient(circle, var(--c-accent) 0%, transparent 70%)" }}
        />
        <div className="relative mx-auto max-w-[1080px] px-5 pb-16 pt-20 text-center">
          <LogoMark className="mx-auto h-14 w-14" />
          <span className="pill pill-accent mt-5 inline-flex">AI video understanding</span>
          <h1 className="mx-auto mt-4 max-w-[760px] text-[clamp(2.1rem,5vw,3.4rem)] font-extrabold leading-[1.06] tracking-[-0.03em] text-ink">
            Understand any video
            <br />
            <span className="text-accent">without watching it</span>
          </h1>
          <p className="mx-auto mt-5 max-w-[620px] text-[1.03rem] leading-relaxed text-ink-muted">
            Paste a link to a lecture, a tutorial or a three-hour practice
            session. You get it taught back to you section by section, with the
            code, the questions it solves, and an assistant that cites the
            moment it got each answer.
          </p>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <button onClick={openSignIn} className="btn btn-primary !px-6 !py-2.5 !text-[0.95rem]">
              Sign in to start
              <ArrowRight className="h-4 w-4" />
            </button>
            <a href="#how" className="btn !px-5 !py-2.5 no-underline">
              How it works
            </a>
          </div>
          <p className="mt-4 text-[0.8rem] text-ink-dim">
            A code to your email — no password to invent. Free plan covers
            videos up to 4 hours.
          </p>
        </div>
      </section>

      {/* ---------------- measured, not claimed ---------------- */}
      <section className="border-b border-border bg-surface/40">
        <div className="mx-auto grid max-w-[1080px] gap-px bg-border px-0 sm:grid-cols-2 lg:grid-cols-4">
          {EVIDENCE.map((e) => (
            <div key={e.label} className="bg-bg px-5 py-7 text-center">
              <p className="font-mono text-[1.75rem] font-bold tabular-nums tracking-tight text-accent">
                {e.figure}
              </p>
              <p className="mt-1 text-[0.86rem] font-semibold text-ink">{e.label}</p>
              <p className="mt-1.5 text-[0.74rem] leading-relaxed text-ink-dim">{e.note}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---------------- what you get ---------------- */}
      <section className="mx-auto max-w-[1080px] px-5 py-16">
        <h2 className="text-center text-[1.65rem] font-extrabold tracking-[-0.02em] text-ink">
          What comes out
        </h2>
        <p className="mx-auto mt-2.5 max-w-[560px] text-center text-[0.92rem] leading-relaxed text-ink-muted">
          One run produces all of this. Nothing here is generated a second time
          from scratch — it is all derived from the same reading of the video.
        </p>

        <div className="mt-9 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {WHAT_YOU_GET.map((f) => (
            <div key={f.title} className="card p-5">
              <span className={`mb-3 grid h-9 w-9 place-items-center rounded-lg ${f.tone}`}>
                <f.icon className="h-[17px] w-[17px]" />
              </span>
              <p className="text-[0.95rem] font-bold tracking-tight text-ink">{f.title}</p>
              <p className="mt-1.5 text-[0.85rem] leading-relaxed text-ink-muted">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---------------- the pipeline ---------------- */}
      <section id="how" className="border-y border-border bg-surface/40">
        <div className="mx-auto max-w-[1080px] px-5 py-16">
          <h2 className="text-center text-[1.65rem] font-extrabold tracking-[-0.02em] text-ink">
            Seven stages
          </h2>
          <p className="mx-auto mt-2.5 max-w-[600px] text-center text-[0.92rem] leading-relaxed text-ink-muted">
            The pipeline aims at a wall-clock budget rather than a fixed set of
            models: it measures the machine, reads the video&apos;s length, and
            picks the heaviest configuration predicted to finish in time.
          </p>

          <ol className="mt-9 grid list-none gap-3 p-0 md:grid-cols-2 lg:grid-cols-4">
            {STAGES.map((s) => (
              <li key={s.n} className="card p-4">
                <span className="font-mono text-[0.7rem] font-bold tracking-wide text-accent">
                  {s.n}
                </span>
                <p className="mt-0.5 text-[0.92rem] font-bold tracking-tight text-ink">{s.name}</p>
                <p className="mt-1.5 text-[0.82rem] leading-relaxed text-ink-muted">{s.body}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ---------------- the plans ---------------- */}
      <section className="mx-auto max-w-[1080px] px-5 py-16">
        <h2 className="text-center text-[1.65rem] font-extrabold tracking-[-0.02em] text-ink">
          What a plan covers
        </h2>
        <p className="mx-auto mt-2.5 max-w-[580px] text-center text-[0.92rem] leading-relaxed text-ink-muted">
          The allowance is counted in model calls rather than in videos,
          because a four-hour video costs about five times a half-hour one and
          charging them alike would be wrong in both directions.
        </p>

        <div className="mx-auto mt-9 grid max-w-[720px] gap-4 sm:grid-cols-2">
          <div className="card p-6">
            <p className="text-[0.72rem] font-bold uppercase tracking-wide text-ink-dim">Free</p>
            <p className="mt-1.5 text-[1.6rem] font-extrabold tracking-tight text-ink">₹0</p>
            <ul className="mt-4 flex list-none flex-col gap-2 p-0 text-[0.86rem] text-ink-muted">
              <li>Videos up to <span className="font-semibold text-ink">4 hours</span></li>
              <li>40 model calls a day</li>
              <li>Every feature, none held back</li>
              <li>No card, nothing to cancel</li>
            </ul>
            <button onClick={openSignIn} className="btn btn-primary mt-5 w-full">
              Start here
            </button>
          </div>

          <div className="card relative overflow-hidden border-accent/30 p-6">
            <span className="pill pill-accent absolute right-4 top-4">For long videos</span>
            <p className="text-[0.72rem] font-bold uppercase tracking-wide text-ink-dim">Premium</p>
            <p className="mt-1.5 text-[1.6rem] font-extrabold tracking-tight text-ink">
              ₹199
              <span className="text-[0.86rem] font-semibold text-ink-dim"> / 30 days</span>
            </p>
            <ul className="mt-4 flex list-none flex-col gap-2 p-0 text-[0.86rem] text-ink-muted">
              <li>Videos up to <span className="font-semibold text-ink">15 hours</span></li>
              <li>400 model calls a day</li>
              <li>An eleven-hour course in one run</li>
              <li>Upgrade from inside the app</li>
            </ul>
            <button onClick={openSignIn} className="btn mt-5 w-full">
              Sign in first
            </button>
          </div>
        </div>
      </section>

      {/* ---------------- close ---------------- */}
      <section className="border-t border-border">
        <div className="mx-auto max-w-[1080px] px-5 py-16 text-center">
          <Gauge className="mx-auto h-7 w-7 text-accent" />
          <h2 className="mt-3.5 text-[1.5rem] font-extrabold tracking-[-0.02em] text-ink">
            Start with one video
          </h2>
          <p className="mx-auto mt-2.5 max-w-[500px] text-[0.92rem] leading-relaxed text-ink-muted">
            A lecture you have been putting off, or a tutorial you only need
            one answer from. It takes a link and a few minutes.
          </p>
          <button onClick={openSignIn} className="btn btn-primary mt-6 !px-6 !py-2.5">
            <Sparkles className="h-4 w-4" />
            Sign in to start
          </button>
        </div>

        <footer className="border-t border-border px-5 py-7">
          <div className="mx-auto flex max-w-[1080px] flex-wrap items-center justify-between gap-3 text-[0.78rem] text-ink-dim">
            <span className="flex items-center gap-2">
              <Network className="h-3.5 w-3.5" />
              Built as a final-year project — the pipeline, the scheduler and
              the evaluation are all in the repository.
            </span>
            <Link href="/app" className="no-underline hover:text-ink">
              Open the app →
            </Link>
          </div>
        </footer>
      </section>
    </div>
  );
}
