"use client";

import { Square } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, humanDuration, type JobView } from "@/lib/api";

const PILL: Record<string, string> = {
  done: "pill-success",
  failed: "pill-danger",
  cancelled: "pill-warning",
};

export default function Library() {
  const [jobs, setJobs] = useState<JobView[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [stopping, setStopping] = useState<string | null>(null);

  useEffect(() => {
    api.recentJobs(50).then(setJobs).catch((e) => setErr(e.message));
  }, []);

  const active = jobs.filter((j) => j.status === "queued" || j.status === "running");

  async function stop(id: string) {
    setStopping(id);
    try {
      const updated = await api.cancelJob(id);
      setJobs((all) => all.map((j) => (j.id === id ? updated : j)));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setStopping(null);
    }
  }

  async function stopAll() {
    setStopping("*");
    try {
      const updated = await Promise.all(active.map((j) => api.cancelJob(j.id)));
      const byId = new Map(updated.map((j) => [j.id, j]));
      setJobs((all) => all.map((j) => byId.get(j.id) ?? j));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setStopping(null);
    }
  }

  return (
    <div className="mx-auto max-w-[820px]">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h1 className="m-0 text-[1.3rem] font-bold text-ink">Library</h1>
        {active.length > 0 && (
          <button
            onClick={stopAll}
            disabled={stopping !== null}
            className="btn gap-1.5 px-3 py-1.5 text-[0.8rem] text-danger hover:bg-danger/[0.08]"
          >
            <Square className="h-3.5 w-3.5" fill="currentColor" strokeWidth={0} />
            {stopping === "*" ? "Stopping…" : `Stop all (${active.length})`}
          </button>
        )}
      </div>

      {err && <p className="text-[0.9rem] text-danger">{err}</p>}
      {jobs.length === 0 && !err && <p className="text-ink-dim">Nothing processed yet.</p>}

      <div className="flex flex-col gap-2">
        {jobs.map((j) => {
          const running = j.status === "queued" || j.status === "running";
          return (
            <div
              key={j.id}
              className="card flex items-center gap-3 p-4 transition-colors hover:border-border-strong"
            >
              <Link
                href={j.status === "done" && j.video_id ? `/v/${j.video_id}` : `/jobs/${j.id}`}
                className="min-w-0 flex-1 no-underline"
              >
                <p className="truncate font-semibold text-ink">{j.title || "Untitled"}</p>
                <p className="mt-0.5 text-[0.78rem] text-ink-dim">
                  {j.video_type} · {j.profile}
                  {j.duration_sec ? ` · ${humanDuration(j.duration_sec)}` : ""}
                  {j.timings?.total ? ` · ran ${humanDuration(j.timings.total)}` : ""}
                </p>
              </Link>

              {running && (
                <button
                  onClick={() => stop(j.id)}
                  disabled={stopping !== null}
                  title="Stop this job"
                  aria-label="Stop this job"
                  className="btn flex-shrink-0 px-2.5 py-1.5 text-danger hover:bg-danger/[0.08]"
                >
                  <Square className="h-3.5 w-3.5" fill="currentColor" strokeWidth={0} />
                </button>
              )}

              <span className={`pill flex-shrink-0 ${PILL[j.status] ?? "pill-accent"}`}>
                {j.status}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
