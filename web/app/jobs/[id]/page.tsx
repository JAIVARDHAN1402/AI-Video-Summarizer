"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useEffect, useRef, useState } from "react";
import { CheckCircle2, Circle, Loader2, Square, XCircle } from "lucide-react";
import { api, hasSessionToken, humanDuration, type JobView } from "@/lib/api";
import { RequireAuth } from "@/components/RequireAuth";

const STAGES: [string, string][] = [
  ["ingest", "Fetch & prepare video"],
  ["analyze", "Transcribe audio + analyze visuals"],
  ["fuse", "Align transcript with scenes"],
  ["summarize", "Explanation & summary"],
  ["index", "Build Q&A index"],
];

function JobPageInner({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [job, setJob] = useState<JobView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const redirected = useRef(false);

  useEffect(() => {
    // Poll unconditionally and let SSE ride on top: the stream makes updates
    // instant when it gets through, but a proxy that buffers text/event-stream
    // (the Next dev rewrite does) would otherwise leave the page stuck on
    // "Starting…" for the whole run — with no way to reach the Stop button.
    let stopped = false;

    const apply = (j: JobView) => {
      if (stopped) return;
      setJob(j);
      if (j.status === "done" && j.video_id && !redirected.current) {
        redirected.current = true;
        setTimeout(() => router.replace(`/v/${j.video_id}`), 700);
      }
      if (j.status === "failed" || j.status === "cancelled") {
        stopped = true;
        clearInterval(poll);
        es?.close();
      }
    };

    // EventSource cannot attach an Authorization header. Authenticated readers
    // still receive the same live state through the deliberately short poll;
    // anonymous/local development retains SSE's instant updates.
    const es = hasSessionToken() ? null : new EventSource(`/api/jobs/${id}/events`);
    if (es) {
      const consume = (e: Event) => {
        try {
          apply(JSON.parse((e as MessageEvent).data) as JobView);
        } catch {
          /* a partial frame — the next poll will catch up */
        }
      };
      es.addEventListener("progress", consume);
      es.addEventListener("end", consume);
      es.onerror = () => es.close();
    }

    const tick = async () => {
      try {
        apply(await api.job(id));
      } catch (err: any) {
        setError(err.message);
      }
    };
    void tick();
    const poll = setInterval(tick, 1500);

    return () => {
      stopped = true;
      es?.close();
      clearInterval(poll);
    };
  }, [id, router]);

  const currentIdx = job
    ? job.stage === "done"
      ? STAGES.length
      : STAGES.findIndex(([k]) => k === job.stage)
    : -1;
  const failed = job?.status === "failed";
  const cancelled = job?.status === "cancelled";
  const active = job?.status === "queued" || job?.status === "running";

  async function stop() {
    if (!job || stopping) return;
    setStopping(true);
    try {
      setJob(await api.cancelJob(job.id));
    } catch (err: any) {
      setError(err.message || "Could not stop this job.");
    } finally {
      setStopping(false);
    }
  }

  return (
    <div className="mx-auto h-full max-w-[760px] overflow-y-auto px-5 py-8">
      <div className="card p-6">
        <div className="mb-1 flex items-center justify-between gap-3">
          <h2 className="m-0 text-[1.1rem] font-semibold">
            {job?.title || "Processing your video"}
          </h2>
          <div className="flex flex-shrink-0 items-center gap-2">
            {job?.routed_worker_class && (
              <span className="pill pill-accent">routed → {job.routed_worker_class}</span>
            )}
            {active && (
              <button
                onClick={stop}
                disabled={stopping}
                className="btn gap-1.5 px-3 py-1.5 text-[0.8rem] text-danger hover:bg-danger/[0.08]"
              >
                <Square className="h-3.5 w-3.5" fill="currentColor" strokeWidth={0} />
                {stopping ? "Stopping…" : "Stop"}
              </button>
            )}
          </div>
        </div>
        <p className="m-0 text-[0.85rem] text-ink-muted">
          {failed
            ? "This run failed."
            : cancelled
              ? "You stopped this run."
              : job?.message || "Starting…"}
        </p>

        <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-surface-raised">
          <div
            className="h-full rounded-full bg-accent transition-[width] duration-500"
            style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }}
          />
        </div>

        <div className="mt-5 flex flex-col gap-0.5">
          {STAGES.map(([key, label], i) => {
            const done = job?.timings?.[key] != null || i < currentIdx || job?.stage === "done";
            const active = i === currentIdx && !failed && !cancelled && job?.stage !== "done";
            const isFail = failed && i === currentIdx;
            return (
              <div
                key={key}
                className="flex items-center gap-3 border-b border-border py-2.5 text-[0.875rem] last:border-0"
              >
                <span className="flex h-[22px] w-[22px] items-center justify-center">
                  {done ? (
                    <CheckCircle2 className="h-[18px] w-[18px] text-success" />
                  ) : isFail ? (
                    <XCircle className="h-[18px] w-[18px] text-danger" />
                  ) : active ? (
                    <Loader2 className="h-[18px] w-[18px] animate-spin text-accent-bright" />
                  ) : (
                    <Circle className="h-[16px] w-[16px] text-ink-dim" />
                  )}
                </span>
                <span className={`flex-1 ${active || done ? "text-ink" : "text-ink-dim"}`}>
                  {label}
                </span>
                <span className="font-mono text-[0.76rem] tabular-nums text-ink-dim">
                  {job?.timings?.[key] != null
                    ? `${job.timings[key].toFixed(1)}s`
                    : active
                      ? "running"
                      : ""}
                </span>
              </div>
            );
          })}
        </div>

        {job && (job.sla_target_sec || job.predicted_sec) && !failed && (
          <p className="mt-4 text-[0.78rem] text-ink-dim">
            {job.predicted_sec != null && `Predicted ~${humanDuration(job.predicted_sec)}`}
            {job.sla_target_sec != null && ` · target ${humanDuration(job.sla_target_sec)}`}
            {job.elapsed_sec != null && ` · elapsed ${humanDuration(job.elapsed_sec)}`}
          </p>
        )}

        {failed && (
          <div className="mt-4 rounded-sm border border-danger/40 bg-danger/[0.1] px-3 py-2 text-[0.85rem] text-danger">
            {job?.error || "Unknown error."}
          </div>
        )}
        {cancelled && (
          <div className="mt-4 flex flex-wrap items-center gap-3 rounded-sm border border-border bg-surface-raised px-3 py-2 text-[0.85rem] text-ink-muted">
            <span>Stopped. Anything already cached is kept, so a re-run resumes from there.</span>
            <Link href="/" className="font-semibold text-accent hover:underline">
              Start another
            </Link>
          </div>
        )}
        {error && (
          <div className="mt-4 rounded-sm border border-danger/40 bg-danger/[0.1] px-3 py-2 text-[0.85rem] text-danger">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

export default function JobPage(props: any) {
  return (
    <RequireAuth what="this run">
      <JobPageInner {...props} />
    </RequireAuth>
  );
}
