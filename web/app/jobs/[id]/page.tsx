"use client";

import { useRouter } from "next/navigation";
import { use, useEffect, useRef, useState } from "react";
import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";
import { api, humanDuration, type JobView } from "@/lib/api";

const STAGES: [string, string][] = [
  ["ingest", "Fetch & prepare video"],
  ["analyze", "Transcribe audio + analyze visuals"],
  ["fuse", "Align transcript with scenes"],
  ["summarize", "Explanation & summary"],
  ["index", "Build Q&A index"],
];

export default function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [job, setJob] = useState<JobView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const redirected = useRef(false);

  useEffect(() => {
    // Prefer SSE; fall back to polling if the stream errors.
    const es = new EventSource(`/api/jobs/${id}/events`);
    let poll: ReturnType<typeof setInterval> | null = null;

    const consume = (data: string) => {
      try {
        const j: JobView = JSON.parse(data);
        setJob(j);
        if (j.status === "done" && j.video_id && !redirected.current) {
          redirected.current = true;
          setTimeout(() => router.replace(`/v/${j.video_id}`), 700);
        }
      } catch {}
    };

    es.addEventListener("progress", (e) => consume((e as MessageEvent).data));
    es.addEventListener("end", (e) => consume((e as MessageEvent).data));
    es.onerror = () => {
      es.close();
      if (!poll) {
        poll = setInterval(async () => {
          try {
            const j = await api.job(id);
            setJob(j);
            if (j.status === "done" && j.video_id && !redirected.current) {
              redirected.current = true;
              clearInterval(poll!);
              router.replace(`/v/${j.video_id}`);
            }
            if (j.status === "failed") clearInterval(poll!);
          } catch (err: any) {
            setError(err.message);
          }
        }, 1500);
      }
    };

    return () => {
      es.close();
      if (poll) clearInterval(poll);
    };
  }, [id, router]);

  const currentIdx = job
    ? job.stage === "done"
      ? STAGES.length
      : STAGES.findIndex(([k]) => k === job.stage)
    : -1;
  const failed = job?.status === "failed";

  return (
    <div className="mx-auto max-w-[720px]">
      <div className="card p-6">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="m-0 text-[1.1rem] font-semibold">
            {job?.title || "Processing your video"}
          </h2>
          {job?.routed_worker_class && (
            <span className="pill pill-accent">routed → {job.routed_worker_class}</span>
          )}
        </div>
        <p className="m-0 text-[0.85rem] text-ink-muted">
          {failed ? "This run failed." : job?.message || "Starting…"}
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
            const active = i === currentIdx && !failed && job?.stage !== "done";
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
        {error && (
          <div className="mt-4 rounded-sm border border-danger/40 bg-danger/[0.1] px-3 py-2 text-[0.85rem] text-danger">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
