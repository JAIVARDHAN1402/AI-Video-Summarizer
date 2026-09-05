"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, humanDuration, type JobView } from "@/lib/api";

export default function Library() {
  const [jobs, setJobs] = useState<JobView[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.recentJobs(50).then(setJobs).catch((e) => setErr(e.message));
  }, []);
  return (
    <div className="mx-auto max-w-[820px]">
      <h1 className="mb-4 text-[1.3rem] font-bold text-ink">Library</h1>
      {err && <p className="text-[0.9rem] text-danger">{err}</p>}
      {jobs.length === 0 && !err && <p className="text-ink-dim">Nothing processed yet.</p>}
      <div className="flex flex-col gap-2">
        {jobs.map((j) => (
          <Link
            key={j.id}
            href={j.status === "done" && j.video_id ? `/v/${j.video_id}` : `/jobs/${j.id}`}
            className="card flex items-center justify-between p-4 no-underline transition-colors hover:border-border-strong"
          >
            <div className="min-w-0">
              <p className="truncate font-semibold text-ink">{j.title || "Untitled"}</p>
              <p className="mt-0.5 text-[0.78rem] text-ink-dim">
                {j.video_type} · {j.profile}
                {j.duration_sec ? ` · ${humanDuration(j.duration_sec)}` : ""}
                {j.timings?.total ? ` · ran ${humanDuration(j.timings.total)}` : ""}
              </p>
            </div>
            <span
              className={`pill ${
                j.status === "done" ? "pill-success" : j.status === "failed" ? "pill-danger" : "pill-accent"
              }`}
            >
              {j.status}
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
