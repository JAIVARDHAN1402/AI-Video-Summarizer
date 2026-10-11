"use client";

import { Film, Search, Sparkles, Square } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { api, humanDuration, type JobView, type VideoSummary } from "@/lib/api";

export default function Library() {
  const [videos, setVideos] = useState<VideoSummary[] | null>(null);
  const [jobs, setJobs] = useState<JobView[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [stopping, setStopping] = useState<string | null>(null);
  const [q, setQ] = useState("");

  useEffect(() => {
    // Two sources, deliberately: the cache says what has been processed, the
    // jobs table says what is happening right now. A processed video with no
    // job row (Streamlit, an older run, a moved cache) must still appear.
    api.videos().then(setVideos).catch((e) => setErr(e.message));
    api.recentJobs(50).then(setJobs).catch(() => undefined);
  }, []);

  const active = jobs.filter((j) => j.status === "queued" || j.status === "running");
  const failed = jobs.filter((j) => j.status === "failed");

  const shown = useMemo(() => {
    const list = videos ?? [];
    const needle = q.trim().toLowerCase();
    const out = needle ? list.filter((v) => v.title.toLowerCase().includes(needle)) : list;
    return [...out].sort((a, b) => b.mtime - a.mtime);
  }, [videos, q]);

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
    <div className="mx-auto h-full max-w-[1000px] overflow-y-auto px-5 py-8">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="m-0 text-[1.45rem] font-bold tracking-tight text-ink">Library</h1>
          <p className="mt-1 text-[0.85rem] text-ink-dim">
            {videos === null
              ? "Reading the cache…"
              : `${videos.length} processed video${videos.length === 1 ? "" : "s"}`}
          </p>
        </div>
        <div className="relative w-full sm:w-[280px]">
          <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-dim" />
          <input
            className="input !pl-9"
            placeholder="Search by title…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
      </div>

      {err && <p className="mb-4 text-[0.9rem] text-danger">{err}</p>}

      {active.length > 0 && (
        <section className="mb-6">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="m-0 text-[0.78rem] font-bold uppercase tracking-wide text-ink-dim">
              In progress
            </h2>
            <button
              onClick={stopAll}
              disabled={stopping !== null}
              className="btn gap-1.5 !px-3 !py-1.5 !text-[0.78rem] text-danger hover:bg-danger/[0.08]"
            >
              <Square className="h-3 w-3" fill="currentColor" strokeWidth={0} />
              {stopping === "*" ? "Stopping…" : `Stop all (${active.length})`}
            </button>
          </div>
          <div className="flex flex-col gap-2">
            {active.map((j) => (
              <div key={j.id} className="card flex items-center gap-3 p-3.5">
                <Link href={`/jobs/${j.id}`} className="min-w-0 flex-1 no-underline">
                  <p className="truncate font-semibold text-ink">{j.title || "Processing…"}</p>
                  <p className="mt-1 truncate text-[0.78rem] text-ink-dim">
                    {j.stage} &middot; {Math.round(j.progress * 100)}% &middot; {j.message}
                  </p>
                  <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-active">
                    <div
                      className="h-full rounded-full bg-accent transition-all"
                      style={{ width: `${Math.max(j.progress * 100, 2)}%` }}
                    />
                  </div>
                </Link>
                <button
                  onClick={() => stop(j.id)}
                  disabled={stopping !== null}
                  title="Stop this job"
                  aria-label="Stop this job"
                  className="btn flex-shrink-0 !px-2.5 !py-1.5 text-danger hover:bg-danger/[0.08]"
                >
                  <Square className="h-3.5 w-3.5" fill="currentColor" strokeWidth={0} />
                </button>
              </div>
            ))}
          </div>
        </section>
      )}

      {videos === null ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="card p-4">
              <div className="skeleton h-4 w-3/4" />
              <div className="skeleton mt-2 h-3 w-1/2" />
            </div>
          ))}
        </div>
      ) : shown.length === 0 ? (
        <div className="card grid place-items-center p-12 text-center">
          <Film className="mb-3 h-7 w-7 text-ink-dim" />
          <p className="font-semibold text-ink">
            {q ? "Nothing matches that search" : "Nothing processed yet"}
          </p>
          {!q && (
            <Link href="/" className="mt-3 no-underline">
              <span className="btn btn-primary">Summarize a video</span>
            </Link>
          )}
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map((v) => (
            <Link
              key={v.video_id}
              href={`/v/${v.video_id}`}
              className="card flex flex-col gap-2 p-4 no-underline transition-all hover:border-border-strong hover:shadow-card-hover"
            >
              <p className="line-clamp-2 font-semibold leading-snug text-ink">{v.title}</p>
              <div className="mt-auto flex flex-wrap gap-1.5 pt-1">
                {v.duration_sec ? (
                  <span className="pill">{humanDuration(v.duration_sec)}</span>
                ) : null}
                {v.has_understanding && (
                  <span className="pill pill-accent gap-1">
                    <Sparkles className="h-3 w-3" />
                    explained
                  </span>
                )}
                {v.has_highlight && <span className="pill">reel</span>}
                {v.processed_sec ? (
                  <span className="pill pill-success">in {humanDuration(v.processed_sec)}</span>
                ) : null}
              </div>
            </Link>
          ))}
        </div>
      )}

      {failed.length > 0 && (
        <details className="mt-7">
          <summary className="cursor-pointer text-[0.8rem] font-semibold text-ink-dim">
            {failed.length} failed run{failed.length === 1 ? "" : "s"}
          </summary>
          <div className="mt-2 flex flex-col gap-2">
            {failed.map((j) => (
              <Link
                key={j.id}
                href={`/jobs/${j.id}`}
                className="card p-3 text-[0.82rem] no-underline"
              >
                <p className="truncate font-medium text-ink">{j.title || "Untitled"}</p>
                <p className="mt-0.5 truncate text-ink-dim">{j.error || "failed"}</p>
              </Link>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
