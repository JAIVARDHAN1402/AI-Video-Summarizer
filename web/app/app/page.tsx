"use client";

import {
  ArrowRight,
  AudioLines,
  Clock,
  Gauge,
  History,
  Layers,
  ListChecks,
  Network,
  Sparkles,
  Upload,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  api,
  hasSessionToken,
  humanDuration,
  mmss,
  quotaRefusalOf,
  type Preflight,
  type Profile,
  type QuotaRefusal,
  type VideoSummary,
  type VideoType,
} from "@/lib/api";
import { RequireAuth } from "@/components/RequireAuth";

const VIDEO_TYPES: { value: VideoType; label: string }[] = [
  { value: "lecture", label: "Lecture / tutorial" },
  { value: "coding", label: "Coding / algorithms" },
  { value: "meeting", label: "Meeting recording" },
  { value: "interview", label: "Interview / podcast" },
  { value: "talk", label: "Conference talk" },
  { value: "news", label: "News segment" },
  { value: "documentary", label: "Documentary / explainer" },
  { value: "other", label: "Other" },
];

const PROFILES: { value: Profile; label: string; hint: string }[] = [
  { value: "fast", label: "Fast", hint: "Quick pass, lighter analysis" },
  { value: "balanced", label: "Balanced", hint: "Full explanation, ~5 min target" },
  { value: "accurate", label: "Accurate", hint: "Largest models, no time limit" },
];

const CAPABILITIES = [
  { icon: Sparkles, label: "Teachable explanation", hint: "section by section, with timestamps" },
  { icon: ListChecks, label: "Quiz", hint: "written from what the video teaches" },
  { icon: AudioLines, label: "Audio overview", hint: "two voices, about five minutes" },
  { icon: Network, label: "Mind map", hint: "the chapter structure at a glance" },
];

function HomeInner() {
  const router = useRouter();
  const [mode, setMode] = useState<"url" | "upload">("url");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [videoType, setVideoType] = useState<VideoType>("lecture");
  const [topic, setTopic] = useState("");
  const [profile, setProfile] = useState<Profile>("balanced");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /**
   * A refusal the server explained — the video is longer than the plan
   * covers, or the day's calls are spent. Kept apart from `error` because
   * there is something to do about it, and printing the JSON it arrived in
   * is not it.
   */
  const [refused, setRefused] = useState<QuotaRefusal | null>(null);
  /**
   * Set when the link has been here before. Submitting used to resume from
   * the cache without saying so, and the run that followed reported 17
   * seconds for a three-hour video — true of that run, and useless as a
   * measurement. The choice is the reader's now.
   */
  const [found, setFound] = useState<Preflight | null>(null);
  const [recent, setRecent] = useState<VideoSummary[] | null>(null);
  const [loginRequired, setLoginRequired] = useState(false);

  // The page is also the way back into finished work, so it should never be a
  // dead end for someone who has processed something before.
  useEffect(() => {
    api
      .videos()
      .then((v) => setRecent([...v].sort((a, b) => b.mtime - a.mtime).slice(0, 6)))
      .catch(() => setRecent([]));
  }, []);

  const ready = mode === "url" ? !!url.trim() : !!file;

  /** Stop before preflight, upload, or a job is created — never after work starts. */
  async function requiresSignIn(): Promise<boolean> {
    try {
      const methods = await api.authMethods();
      if (methods.auth_required && !hasSessionToken()) {
        setLoginRequired(true);
        return true;
      }
    } catch {
      // The eventual request retains its normal, useful network error.
    }
    return false;
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!ready) return;
    setError(null);
    if (await requiresSignIn()) return;
    // Uploads get a fresh id every time, so only a pasted link can collide.
    if (mode === "url") {
      setBusy(true);
      try {
        const pre = await api.preflight({ url: url.trim() });
        if (pre.known) {
          setFound(pre);
          setBusy(false);
          return;
        }
      } catch {
        // Not knowing is not a reason to refuse to start; fall through.
      }
    }
    await start(false);
  }

  async function start(force: boolean) {
    if (await requiresSignIn()) return;
    setFound(null);
    setBusy(true);
    setError(null);
    try {
      const common = { video_type: videoType, topic: topic.trim(), profile, force };
      let job;
      if (mode === "upload" && file) {
        const { upload_id } = await api.upload(file);
        job = await api.createJob({ upload_id, ...common });
      } else {
        job = await api.createJob({ url: url.trim(), ...common });
      }
      router.push(`/jobs/${job.id}`);
    } catch (err: any) {
      const refusal = quotaRefusalOf(err);
      if (refusal) setRefused(refusal);
      else setError(err.message || "Could not start the job.");
      setBusy(false);
    }
  }

  return (
    // The shell no longer scrolls the window, so this page owns its scroll.
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1080px] px-5 pb-20 pt-10">
        <header className="mb-6 max-w-[620px]">
          <h1 className="text-[1.75rem] font-extrabold leading-[1.15] tracking-[-0.02em] text-ink">
            Summarize a video
          </h1>
          <p className="mt-2 text-[0.92rem] leading-relaxed text-ink-muted">
            Paste a link or upload a file. Longer videos take longer — the
            estimate appears before anything starts.
          </p>
        </header>

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          {/* ---------------- the form ---------------- */}
          <form onSubmit={submit} className="card p-5">
            <div className="mb-4 flex w-fit gap-0.5 rounded-sm border border-border bg-bg-elevated p-1">
              {(["url", "upload"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={`rounded-xs px-3.5 py-1.5 text-[0.82rem] font-semibold transition-colors ${
                    mode === m ? "bg-surface text-ink shadow-card" : "text-ink-dim hover:text-ink"
                  }`}
                >
                  {m === "url" ? "Paste a link" : "Upload a file"}
                </button>
              ))}
            </div>

            {mode === "url" ? (
              <div className="mb-3.5">
                <label className="label" htmlFor="url">
                  Video URL
                </label>
                <input
                  id="url"
                  className="input"
                  placeholder="https://www.youtube.com/watch?v=…"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
              </div>
            ) : (
              <div className="mb-3.5">
                <label className="label" htmlFor="file">
                  Video file
                </label>
                <label
                  htmlFor="file"
                  className="flex cursor-pointer items-center gap-2.5 rounded-sm border border-dashed border-border-strong bg-surface-raised px-3.5 py-4 text-[0.88rem] text-ink-muted transition-colors hover:border-accent hover:text-ink"
                >
                  <Upload className="h-4 w-4 flex-shrink-0" />
                  <span className="min-w-0 truncate">
                    {file ? file.name : "Choose a video file…"}
                  </span>
                </label>
                <input
                  id="file"
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </div>
            )}

            <div className="mb-3.5 grid gap-3.5 sm:grid-cols-2">
              <div>
                <label className="label" htmlFor="type">
                  What kind of video is this?
                </label>
                <select
                  id="type"
                  className="select"
                  value={videoType}
                  onChange={(e) => setVideoType(e.target.value as VideoType)}
                >
                  {VIDEO_TYPES.map((t) => (
                    <option key={t.value} value={t.value}>
                      {t.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label" htmlFor="topic">
                  Topic (optional)
                </label>
                <input
                  id="topic"
                  className="input"
                  placeholder="e.g. Backtracking, Q3 roadmap"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                />
              </div>
            </div>

            <div className="mb-1">
              <span className="label">Processing profile</span>
              <div className="grid gap-2 sm:grid-cols-3">
                {PROFILES.map((p) => (
                  <button
                    key={p.value}
                    type="button"
                    onClick={() => setProfile(p.value)}
                    className={`rounded-lg border px-3 py-2.5 text-left transition-all ${
                      profile === p.value
                        ? "border-accent bg-accent/[0.07]"
                        : "border-border bg-surface-raised hover:border-border-strong"
                    }`}
                  >
                    <span className="block text-[0.85rem] font-semibold text-ink">{p.label}</span>
                    <span className="mt-0.5 block text-[0.7rem] leading-snug text-ink-dim">
                      {p.hint}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {error && <p className="mt-3 text-[0.85rem] text-danger">{error}</p>}

            {refused && <OverPlan refusal={refused} onDismiss={() => setRefused(null)} />}

            {found && <AlreadyHere found={found} onPick={start} onCancel={() => setFound(null)} />}

            <button className="btn btn-primary mt-4 w-full gap-2" disabled={busy || !ready}>
              {busy ? "Starting…" : "Summarize video"}
              {!busy && <ArrowRight className="h-4 w-4" />}
            </button>
          </form>

          {/* ---------------- what you get, and what you already have ------- */}
          <div className="flex flex-col gap-5">
            <section className="card p-5">
              <h2 className="m-0 mb-3 text-[0.72rem] font-bold uppercase tracking-wide text-ink-dim">
                What comes out
              </h2>
              <div className="flex flex-col gap-2.5">
                {CAPABILITIES.map((c) => (
                  <div key={c.label} className="flex items-start gap-2.5">
                    <span className="tile-icon bg-accent/[0.12] text-accent">
                      <c.icon className="h-[15px] w-[15px]" />
                    </span>
                    <span className="min-w-0">
                      <span className="block text-[0.85rem] font-semibold text-ink">
                        {c.label}
                      </span>
                      <span className="block text-[0.74rem] text-ink-dim">{c.hint}</span>
                    </span>
                  </div>
                ))}
              </div>
            </section>

            <section className="card flex min-h-0 flex-col p-5">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="m-0 text-[0.72rem] font-bold uppercase tracking-wide text-ink-dim">
                  Recently processed
                </h2>
                <Link
                  href="/library"
                  className="text-[0.76rem] font-medium text-accent no-underline hover:underline"
                >
                  Library
                </Link>
              </div>

              {recent === null ? (
                <div className="flex flex-col gap-2">
                  {[0, 1, 2].map((i) => (
                    <div key={i} className="skeleton h-5 w-full" />
                  ))}
                </div>
              ) : recent.length === 0 ? (
                <p className="text-[0.82rem] text-ink-dim">
                  Nothing yet &mdash; the first video you process appears here.
                </p>
              ) : (
                <ul className="m-0 flex list-none flex-col gap-0.5 p-0">
                  {recent.map((v) => (
                    <li key={v.video_id}>
                      <Link
                        href={`/v/${v.video_id}`}
                        className="flex items-center gap-2 rounded-md px-1.5 py-1.5 no-underline transition-colors hover:bg-surface-hover"
                      >
                        <span className="min-w-0 flex-1 truncate text-[0.82rem] text-ink">
                          {v.title}
                        </span>
                        {v.has_understanding && (
                          <Sparkles className="h-3 w-3 flex-shrink-0 text-accent" />
                        )}
                        {v.duration_sec ? (
                          <span className="flex flex-shrink-0 items-center gap-1 font-mono text-[0.68rem] tabular-nums text-ink-dim">
                            <Clock className="h-3 w-3" />
                            {humanDuration(v.duration_sec)}
                          </span>
                        ) : null}
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <Link
              href="/notebooks"
              className="card flex items-center gap-3 p-4 no-underline transition-all hover:border-border-strong hover:shadow-card-hover"
            >
              <span className="tile-icon bg-violet/[0.12] text-violet">
                <Layers className="h-[15px] w-[15px]" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-[0.85rem] font-semibold text-ink">
                  Ask across several videos
                </span>
                <span className="block text-[0.74rem] leading-snug text-ink-dim">
                  Group a playlist or a module into a notebook and question all of it at once.
                </span>
              </span>
              <ArrowRight className="h-4 w-4 flex-shrink-0 text-ink-dim" />
            </Link>
          </div>
        </div>
      </div>
      {loginRequired && (
        <div className="fixed inset-0 z-[60] grid place-items-center bg-ink/40 px-4" role="dialog" aria-modal="true" aria-labelledby="login-gate-title">
          <section className="card w-full max-w-[420px] p-6 shadow-card-hover">
            <h2 id="login-gate-title" className="m-0 text-xl font-bold text-ink">Sign in to process a video</h2>
            <p className="mt-3 text-[0.9rem] leading-relaxed text-ink-muted">
              Your account keeps processing limits fair and lets us track your daily allowance. Sign in first, then submit your link or upload.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  setLoginRequired(false);
                  window.dispatchEvent(new Event("vs-open-auth"));
                }}
              >
                Sign in now
              </button>
              <button type="button" className="btn" onClick={() => setLoginRequired(false)}>Not now</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

const STAGE_LABEL: Record<string, string> = {
  ingest: "Fetch & prepare",
  analyze: "Transcribe + visuals",
  fuse: "Align transcript",
  summarize: "Explanation",
  index: "Q&A index",
};

/**
 * Shown when the pasted link has been processed here before.
 *
 * Resuming is usually what someone wants after a failure — the hours of
 * transcription are already paid for. But it used to happen silently, and
 * then the run report called a seventeen-second resume "624x faster than
 * watching". Both choices are offered, and each says plainly what it costs.
 */
function AlreadyHere({
  found,
  onPick,
  onCancel,
}: {
  found: Preflight;
  onPick: (force: boolean) => void;
  onCancel: () => void;
}) {
  const stages = Object.entries(found.stages);
  const done = stages.filter(([, ok]) => ok);
  return (
    <div className="mt-4 rounded-lg border border-warning/30 bg-warning/[0.06] p-4">
      <p className="flex items-center gap-1.5 text-[0.84rem] font-semibold text-warning">
        <History className="h-4 w-4" />
        This link has been processed here before
      </p>
      <p className="mt-1.5 text-[0.82rem] leading-relaxed text-ink-muted">
        {found.title ? <span className="text-ink">{found.title}</span> : "This video"}
        {found.duration_sec ? ` · ${humanDuration(found.duration_sec)}` : ""} ·{" "}
        {found.stages_done} of {found.stages_total} stages already done
        {found.last_error ? " · the last run failed" : ""}
        {found.previous_total_sec != null
          ? ` · the last run took ${mmss(found.previous_total_sec)}`
          : ""}
      </p>

      <ul className="mt-2.5 flex flex-wrap gap-1.5">
        {stages.map(([name, ok]) => (
          <li key={name} className={`pill ${ok ? "pill-success" : ""}`}>
            {STAGE_LABEL[name] || name}
            {ok ? " ✓" : ""}
          </li>
        ))}
      </ul>

      <div className="mt-3.5 flex flex-wrap gap-2">
        <button type="button" onClick={() => onPick(false)} className="btn btn-primary">
          Resume — reuse {done.length} finished {done.length === 1 ? "stage" : "stages"}
        </button>
        <button type="button" onClick={() => onPick(true)} className="btn">
          Process fresh — redo everything
        </button>
        <button type="button" onClick={onCancel} className="btn">
          Cancel
        </button>
      </div>
      <p className="mt-2 text-[0.74rem] leading-relaxed text-ink-dim">
        Resuming is faster but its timings describe only the stages that
        actually ran. Process fresh when the number itself is the point — a
        measurement, a demo, or a report.
      </p>
    </div>
  );
}

/**
 * Shown when the server refused the job because of the reader's plan.
 *
 * The refusal already carries everything needed to explain itself — how long
 * the video is, what the plan covers, how many calls are left and when they
 * come back. Printing the JSON it arrived in told the reader nothing they
 * could act on.
 */
function OverPlan({
  refusal,
  onDismiss,
}: {
  refusal: QuotaRefusal;
  onDismiss: () => void;
}) {
  const tooLong = refusal.reason === "video_too_long";
  return (
    <div className="mt-4 rounded-lg border border-warning/30 bg-warning/[0.06] p-4">
      <p className="flex items-center gap-1.5 text-[0.84rem] font-semibold text-warning">
        <Gauge className="h-4 w-4" />
        {tooLong ? "Longer than your plan covers" : "Today's processing is used up"}
      </p>

      <p className="mt-1.5 text-[0.85rem] leading-relaxed text-ink">{refusal.message}</p>

      {tooLong ? (
        <p className="mt-1.5 text-[0.8rem] leading-relaxed text-ink-muted">
          Premium covers videos up to 15 hours. A shorter video, or a clip of this one, works
          on the free plan right now.
        </p>
      ) : (
        <p className="mt-1.5 text-[0.8rem] leading-relaxed text-ink-muted">
          The allowance is counted in model calls rather than videos, because a four-hour
          video costs about five times a half-hour one.
          {refusal.resets_at
            ? ` It comes back at ${new Date(refusal.resets_at).toLocaleString()}.`
            : ""}
        </p>
      )}

      {!tooLong && typeof refusal.daily_calls === "number" && (
        <div className="mt-3">
          <div className="h-1.5 overflow-hidden rounded-full bg-surface-hover">
            <div
              className="h-full rounded-full bg-warning"
              style={{
                width: `${Math.min(100, Math.round(((refusal.used ?? 0) / refusal.daily_calls) * 100))}%`,
              }}
            />
          </div>
          <p className="mt-1.5 font-mono text-[0.72rem] tabular-nums text-ink-dim">
            {refusal.used ?? 0} of {refusal.daily_calls} used · this video needs{" "}
            {refusal.needed ?? "?"}
          </p>
        </div>
      )}

      <div className="mt-3.5 flex flex-wrap gap-2">
        {refusal.tier !== "premium" && (
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              // The upgrade lives in the account menu, which listens for this.
              window.dispatchEvent(new Event("vs-open-auth"));
              onDismiss();
            }}
          >
            See premium
          </button>
        )}
        <button type="button" onClick={onDismiss} className="btn">
          Try another video
        </button>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <RequireAuth what="the summarizer">
      <HomeInner />
    </RequireAuth>
  );
}
