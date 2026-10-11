"use client";

import { Printer } from "lucide-react";
import { useMemo } from "react";
import { humanDuration, mmss } from "@/lib/api";
import { Empty } from "@/components/StudioViews";

/** A one-page visual summary, assembled entirely from artefacts the pipeline
 *  has already written. No model is called: everything here is a different
 *  arrangement of the understanding document, the scene list and the timings,
 *  which is why it appears instantly and works offline. */
export function Infographic({
  title,
  u,
  scenes,
  durationSec,
  timings,
  onSeek,
}: {
  title: string;
  u: any;
  scenes: any[];
  durationSec: number;
  timings: Record<string, number>;
  onSeek: (s: number) => void;
}) {
  const chapters: any[] = useMemo(
    () =>
      [...(u?.chapters || [])]
        .map((c: any) => ({
          title: c.title || "Untitled",
          sec: c.start_sec ?? c.start ?? 0,
        }))
        .sort((a, b) => a.sec - b.sec),
    [u],
  );

  const points: string[] = (u?.key_points || []).slice(0, 6);
  const glossary: any[] = (u?.glossary || []).slice(0, 14);
  const importance = useMemo(
    () => scenes.map((s) => (typeof s.importance === "number" ? s.importance : 0)),
    [scenes],
  );
  const hasImportance = importance.some((v) => v > 0);

  if (!u) {
    return (
      <Empty
        what="an infographic"
        why="this video was processed without a language model, so there is no explanation to lay out"
      />
    );
  }

  const stats: [string, string][] = [
    [humanDuration(durationSec), "Runtime"],
    [String(chapters.length), "Chapters"],
    [String(scenes.length), "Scenes"],
    ...(timings.total
      ? ([[humanDuration(timings.total), "Processed in"]] as [string, string][])
      : []),
    ...(timings.total && durationSec
      ? ([[`${Math.round(durationSec / timings.total)}×`, "Faster than watching"]] as [
          string,
          string,
        ][])
      : []),
  ];

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-shrink-0 items-center justify-between gap-2 border-b border-border p-3 print:hidden">
        <span className="text-[0.8rem] text-ink-dim">
          Built from the explanation &mdash; nothing was generated for this view
        </span>
        <button onClick={() => window.print()} className="btn gap-1.5 !px-3 !py-1.5 !text-[0.78rem]">
          <Printer className="h-3.5 w-3.5" />
          Print / save as PDF
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <article className="mx-auto max-w-[840px] p-8">
          {/* ---- hero ---- */}
          <header className="mb-7">
            <p className="text-[0.7rem] font-bold uppercase tracking-[0.18em] text-accent">
              {u.video_type || "video"} {u.topic ? `· ${u.topic}` : ""}
            </p>
            <h1 className="mt-2 text-[2rem] font-extrabold leading-[1.1] tracking-tight text-ink">
              {title}
            </h1>
            {u.overview && (
              <p className="mt-3 text-[1rem] leading-relaxed text-ink-muted">{u.overview}</p>
            )}
          </header>

          {/* ---- stat band ---- */}
          <div className="mb-8 grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
            {stats.map(([v, l]) => (
              <div
                key={l}
                className="rounded-xl border border-border bg-surface-raised px-3 py-3 text-center"
              >
                <div className="text-[1.3rem] font-extrabold leading-none tabular-nums text-ink">
                  {v}
                </div>
                <div className="mt-1.5 text-[0.62rem] font-bold uppercase tracking-wide text-ink-dim">
                  {l}
                </div>
              </div>
            ))}
          </div>

          {/* ---- importance ribbon: where the video is densest ---- */}
          {hasImportance && (
            <section className="mb-8">
              <h2 className="mb-2 text-[0.72rem] font-bold uppercase tracking-wide text-ink-dim">
                Where the material is densest
              </h2>
              <div className="flex h-16 items-end gap-[2px] rounded-lg border border-border bg-surface-raised p-2">
                {importance.map((v, i) => (
                  <button
                    key={i}
                    onClick={() => onSeek(scenes[i]?.start ?? 0)}
                    title={`${mmss(scenes[i]?.start ?? 0)} — importance ${(v * 100).toFixed(0)}%`}
                    className="min-w-[2px] flex-1 rounded-t-[2px] bg-accent transition-opacity hover:opacity-70"
                    style={{ height: `${Math.max(v * 100, 4)}%` }}
                  />
                ))}
              </div>
              <p className="mt-1.5 text-[0.7rem] text-ink-dim">
                Scene importance across the runtime, scored by the model trained for this project.
                Click a bar to jump there.
              </p>
            </section>
          )}

          {/* ---- key points ---- */}
          {points.length > 0 && (
            <section className="mb-8">
              <h2 className="mb-3 text-[1.05rem] font-bold tracking-tight text-ink">
                What this video teaches
              </h2>
              <div className="grid gap-2.5 sm:grid-cols-2">
                {points.map((p, i) => (
                  <div
                    key={i}
                    className="flex gap-3 rounded-xl border border-border bg-surface-raised p-3.5"
                  >
                    <span className="grid h-6 w-6 flex-shrink-0 place-items-center rounded-lg bg-accent font-mono text-[0.7rem] font-bold text-white">
                      {i + 1}
                    </span>
                    <p className="min-w-0 text-[0.86rem] leading-relaxed text-ink">{p}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* ---- chapter timeline ---- */}
          {chapters.length > 0 && durationSec > 0 && (
            <section className="mb-8">
              <h2 className="mb-3 text-[1.05rem] font-bold tracking-tight text-ink">
                How it is laid out
              </h2>
              <div className="relative mb-2 h-2 overflow-hidden rounded-full bg-surface-active">
                {chapters.map((c, i) => (
                  <span
                    key={i}
                    className="absolute top-0 h-full w-[2px] bg-accent"
                    style={{ left: `${Math.min((c.sec / durationSec) * 100, 99.6)}%` }}
                  />
                ))}
              </div>
              <ol className="m-0 grid list-none gap-x-5 gap-y-1 p-0 sm:grid-cols-2">
                {chapters.map((c, i) => (
                  <li key={i}>
                    <button
                      onClick={() => onSeek(c.sec)}
                      className="flex w-full gap-2 rounded-md px-1 py-1 text-left transition-colors hover:bg-surface-hover"
                    >
                      <span className="font-mono text-[0.72rem] tabular-nums text-accent">
                        {mmss(c.sec)}
                      </span>
                      <span className="min-w-0 flex-1 text-[0.82rem] leading-snug text-ink-muted">
                        {c.title}
                      </span>
                    </button>
                  </li>
                ))}
              </ol>
            </section>
          )}

          {/* ---- vocabulary ---- */}
          {glossary.length > 0 && (
            <section className="mb-8">
              <h2 className="mb-3 text-[1.05rem] font-bold tracking-tight text-ink">
                Vocabulary it assumes
              </h2>
              <div className="flex flex-wrap gap-1.5">
                {glossary.map((g, i) => (
                  <span
                    key={i}
                    title={g.definition}
                    className="rounded-full border border-border bg-surface-raised px-3 py-1.5 text-[0.8rem] font-medium text-ink"
                  >
                    {g.term}
                  </span>
                ))}
              </div>
            </section>
          )}

          <footer className="border-t border-border pt-4 text-[0.7rem] text-ink-dim">
            Generated by the AI Video Summarization System
            {u.model ? ` · understanding by ${u.model}` : ""}
            {u.generated_in_sec ? ` in ${u.generated_in_sec}s` : ""}
          </footer>
        </article>
      </div>
    </div>
  );
}
