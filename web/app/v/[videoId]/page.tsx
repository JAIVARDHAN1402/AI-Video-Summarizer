"use client";

import { use, useEffect, useMemo, useState } from "react";
import { api, humanDuration, type VideoResult } from "@/lib/api";
import { SectionHead, Stamp, VideoPlayer } from "@/components/Bits";
import { UnderstandingView } from "@/components/UnderstandingView";

const TABS = ["Explanation", "Chapters", "Transcript", "Q&A", "Scenes", "Report"] as const;
type Tab = (typeof TABS)[number];

export default function VideoPage({ params }: { params: Promise<{ videoId: string }> }) {
  const { videoId } = use(params);
  const [data, setData] = useState<VideoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Explanation");
  const [seek, setSeek] = useState(0);

  useEffect(() => {
    api.video(videoId).then(setData).catch((e) => setError(e.message));
  }, [videoId]);

  const sourceRef: string = data?.meta?.source_ref || "";
  const onSeek = (s: number) => {
    setSeek(s);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  if (error)
    return (
      <div className="card mx-auto max-w-[560px] p-6 text-[0.9rem] text-danger">{error}</div>
    );
  if (!data)
    return <div className="py-16 text-center text-ink-dim">Loading…</div>;

  const u = data.understanding;
  const scenes = data.scenes?.scenes || [];
  const transcript = data.transcript?.segments || [];
  const timings = (data.state?.timings || {}) as Record<string, number>;
  const dur = data.meta?.duration_sec || 0;

  return (
    <div>
      <div className="mb-4 border-t border-border pt-5">
        <h1 className="mb-2 text-[1.4rem] font-bold leading-tight tracking-tight text-ink">
          {data.meta?.title || "Untitled video"}
        </h1>
        <div className="flex flex-wrap gap-2">
          {dur > 0 && <span className="pill">{humanDuration(dur)}</span>}
          {timings.total && (
            <span className="pill pill-success">
              Processed in {humanDuration(timings.total)}
            </span>
          )}
          {timings.total && dur > 0 && (
            <span className="pill pill-accent">
              {Math.round(dur / timings.total)}× faster than watching
            </span>
          )}
          {u?.model && <span className="pill">{u.model}</span>}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[7fr_5fr]">
        <VideoPlayer sourceRef={sourceRef} startSec={seek} />
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4 lg:grid-cols-2">
          {[
            [String(u?.explanation?.length ?? u?.chapters?.length ?? 0), "Sections"],
            [String(scenes.length), "Scenes"],
            [
              (data.transcript?.segments || [])
                .reduce((n, s) => n + s.text.split(/\s+/).length, 0)
                .toLocaleString(),
              "Words",
            ],
            [humanDuration(dur), "Length"],
          ].map(([v, l]) => (
            <div
              key={l}
              className="rounded border border-border bg-gradient-to-b from-surface-raised to-surface p-3.5"
            >
              <div className="text-[1.35rem] font-bold leading-tight tabular-nums text-ink">
                {v}
              </div>
              <div className="mt-1 text-[0.68rem] font-semibold uppercase tracking-wide text-ink-dim">
                {l}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 mb-4 flex w-fit gap-0.5 overflow-x-auto rounded border border-border bg-surface p-1">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`whitespace-nowrap rounded-xs px-3.5 py-2 text-[0.85rem] font-semibold transition-colors ${
              tab === t
                ? "bg-surface-active text-ink"
                : "text-ink-dim hover:bg-surface-hover hover:text-ink"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "Explanation" &&
        (u ? (
          <UnderstandingView u={u} sourceRef={sourceRef} onSeek={onSeek} />
        ) : (
          <FallbackSummary summary={data.summary} sourceRef={sourceRef} onSeek={onSeek} />
        ))}

      {tab === "Chapters" && (
        <div className="flex flex-col">
          {(u?.chapters || data.summary?.chapters || []).map((c: any, i: number) => (
            <div key={i} className="flex gap-4 border-b border-border py-3.5 last:border-0">
              <Stamp sec={c.start ?? c.start_sec} sourceRef={sourceRef} onSeek={onSeek} />
              <div>
                <p className="font-semibold text-ink">{c.title}</p>
                <p className="mt-0.5 text-[0.87rem] leading-relaxed text-ink-muted">
                  {c.summary}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {tab === "Transcript" && (
        <TranscriptView segments={transcript} sourceRef={sourceRef} onSeek={onSeek} />
      )}

      {tab === "Q&A" && <AskBox videoId={videoId} sourceRef={sourceRef} onSeek={onSeek} />}

      {tab === "Scenes" && (
        <ScenesView
          scenes={scenes}
          highlight={data.highlight}
          sourceRef={sourceRef}
          onSeek={onSeek}
        />
      )}

      {tab === "Report" && <ReportView timings={timings} dur={dur} meta={data.meta} />}
    </div>
  );
}

function FallbackSummary({
  summary,
  sourceRef,
  onSeek,
}: {
  summary: any;
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const sections = (summary?.chunk_summaries || []).filter((s: any) => s.text);
  return (
    <div className="flex flex-col gap-6">
      {sections.length > 0 ? (
        <section>
          <SectionHead title="Walkthrough" hint="Local model — set GEMINI_API_KEY for the full explanation" />
          {sections.map((s: any, i: number) => (
            <div key={i} className="border-b border-border py-3 last:border-0">
              <Stamp sec={s.start} sourceRef={sourceRef} onSeek={onSeek} />
              <p className="prose-body mt-1">{s.text}</p>
            </div>
          ))}
        </section>
      ) : (
        <p className="prose-body">{summary?.overall_summary || "No summary was produced."}</p>
      )}
    </div>
  );
}

function ScenesView({
  scenes,
  highlight,
  sourceRef,
  onSeek,
}: {
  scenes: any[];
  highlight: VideoResult["highlight"];
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const hasImportance = scenes.some((s) => typeof s.importance === "number");
  const [byImportance, setByImportance] = useState(false);
  const ordered = useMemo(() => {
    if (!byImportance) return scenes;
    return [...scenes].sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0));
  }, [scenes, byImportance]);

  return (
    <div className="flex flex-col gap-5">
      {highlight && (
        <section>
          <SectionHead
            title="Highlight reel"
            hint={
              highlight.seconds
                ? `${Math.round(highlight.seconds)}s · ${highlight.scene_ids.length} top scenes`
                : "top scenes by importance"
            }
          />
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            controls
            src={highlight.url}
            className="w-full rounded border border-border bg-black"
          />
        </section>
      )}

      {hasImportance && (
        <div className="flex items-center gap-2 text-[0.8rem]">
          <span className="text-ink-dim">Order</span>
          <div className="flex gap-0.5 rounded border border-border bg-surface p-1">
            {[
              ["Timeline", false],
              ["Importance", true],
            ].map(([label, val]) => (
              <button
                key={label as string}
                onClick={() => setByImportance(val as boolean)}
                className={`rounded-xs px-2.5 py-1 font-semibold transition-colors ${
                  byImportance === val ? "bg-surface-active text-ink" : "text-ink-dim hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3.5 sm:grid-cols-3 lg:grid-cols-4">
        {ordered.map((s: any, i: number) => (
          <div key={i} className="overflow-hidden rounded border border-border bg-surface pb-2">
            {s.keyframe_url && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={s.keyframe_url} alt="" loading="lazy" className="aspect-video w-full object-cover" />
            )}
            {typeof s.importance === "number" && (
              <div className="h-1 w-full bg-surface-raised">
                <div
                  className="h-full bg-gradient-to-r from-accent to-violet"
                  style={{ width: `${Math.round(s.importance * 100)}%` }}
                  title={`importance ${(s.importance * 100).toFixed(0)}%`}
                />
              </div>
            )}
            <div className="mt-2 flex flex-wrap items-center gap-1.5 px-2.5">
              <Stamp sec={s.start} sourceRef={sourceRef} onSeek={onSeek} />
              {(s.objects || []).slice(0, 3).map((o: any, k: number) => (
                <span
                  key={k}
                  className="rounded-xs border border-border bg-surface-raised px-1.5 py-0.5 text-[0.68rem] text-ink-muted"
                >
                  {o.label}
                </span>
              ))}
            </div>
            {s.ocr_text && (
              <p className="mt-1 line-clamp-3 px-2.5 text-[0.72rem] leading-snug text-ink-dim">
                {s.ocr_text}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function TranscriptView({
  segments,
  sourceRef,
  onSeek,
}: {
  segments: { start: number; end: number; text: string }[];
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const [q, setQ] = useState("");
  const filtered = useMemo(
    () =>
      q.trim()
        ? segments.filter((s) => s.text.toLowerCase().includes(q.toLowerCase()))
        : segments,
    [q, segments],
  );
  return (
    <div>
      <input
        className="input mb-3"
        placeholder="Search the transcript…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <div className="max-h-[560px] overflow-y-auto pr-1">
        {filtered.slice(0, 600).map((s, i) => (
          <div key={i} className="flex gap-3.5 rounded-xs px-2 py-1.5 hover:bg-surface">
            <Stamp sec={s.start} sourceRef={sourceRef} onSeek={onSeek} />
            <span className="text-[0.885rem] leading-relaxed text-ink">{s.text}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[0.78rem] text-ink-dim">
        {filtered.length.toLocaleString()} of {segments.length.toLocaleString()} lines
        {filtered.length > 600 && " — showing the first 600"}
      </p>
    </div>
  );
}

function AskBox({
  videoId,
  sourceRef,
  onSeek,
}: {
  videoId: string;
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<
    { q: string; a: string; confidence: string; sources: any[] }[]
  >([]);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim() || busy) return;
    const question = q.trim();
    setQ("");
    setBusy(true);
    try {
      const r = await api.ask(videoId, question);
      setHistory((h) => [...h, { q: question, a: r.answer, confidence: r.confidence, sources: r.sources }]);
    } catch (err: any) {
      setHistory((h) => [
        ...h,
        { q: question, a: `Error: ${err.message}`, confidence: "low", sources: [] },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      {history.length === 0 && (
        <p className="prose-body mb-4 text-ink-muted">
          Ask anything about the video. Every answer cites the timestamps it came from.
        </p>
      )}
      <div className="mb-4 flex flex-col gap-3">
        {history.map((h, i) => (
          <div key={i} className="rounded border border-border bg-surface p-4">
            <p className="mb-2 font-semibold text-ink">{h.q}</p>
            <span
              className={`mb-2 inline-block ${
                h.confidence === "high" ? "pill pill-success" : "pill pill-warning"
              }`}
            >
              {h.confidence} confidence
            </span>
            <p className="prose-body">{h.a}</p>
            {h.sources?.length > 0 && (
              <details className="mt-2">
                <summary className="cursor-pointer text-[0.8rem] text-ink-dim">
                  Sources ({h.sources.length})
                </summary>
                <div className="mt-2 flex flex-col gap-2">
                  {h.sources.map((s: any, k: number) => (
                    <div key={k} className="flex gap-2 text-[0.83rem] text-ink-muted">
                      <Stamp sec={s.start} sourceRef={sourceRef} onSeek={onSeek} />
                      <span className="line-clamp-3">{s.text}</span>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        ))}
      </div>
      <form onSubmit={ask} className="flex gap-2">
        <input
          className="input"
          placeholder="Ask a question about this video…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <button className="btn btn-primary" disabled={busy || !q.trim()}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
    </div>
  );
}

function ReportView({
  timings,
  dur,
  meta,
}: {
  timings: Record<string, number>;
  dur: number;
  meta: any;
}) {
  const plan = meta?.plan || {};
  const total = timings.total;
  const longest = Math.max(...Object.entries(timings).filter(([k]) => k !== "total").map(([, v]) => v), 1);
  return (
    <div className="flex flex-col gap-6">
      {total && dur > 0 && (
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          {[
            [humanDuration(total), "Total time"],
            [`${Math.round(dur / total)}×`, "Faster than watching"],
            [(total / dur).toFixed(2), "Sec per video-sec"],
            [humanDuration(dur), "Video length"],
          ].map(([v, l]) => (
            <div key={l} className="rounded border border-border bg-surface-raised p-3.5">
              <div className="text-[1.3rem] font-bold tabular-nums text-ink">{v}</div>
              <div className="mt-1 text-[0.68rem] font-semibold uppercase tracking-wide text-ink-dim">
                {l}
              </div>
            </div>
          ))}
        </div>
      )}
      <div>
        <SectionHead title="Time per stage" />
        {Object.entries(timings)
          .filter(([k]) => k !== "total")
          .map(([k, v]) => (
            <div key={k} className="mb-2">
              <div className="mb-1 flex justify-between text-[0.85rem]">
                <span className="capitalize text-ink">{k}</span>
                <span className="font-mono tabular-nums text-ink-muted">{v.toFixed(1)}s</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-raised">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-accent to-violet"
                  style={{ width: `${Math.min((v / longest) * 100, 100)}%` }}
                />
              </div>
            </div>
          ))}
      </div>
      {Object.keys(plan).length > 0 && (
        <div>
          <SectionHead title="Configuration chosen for this video" />
          <div className="flex flex-wrap gap-2">
            {[
              `ASR: ${plan.asr_model}`,
              `beam ${plan.asr_beam_size}`,
              `${plan.asr_workers} workers`,
              `Summarizer: ${String(plan.summarizer_model || "").split("/").pop()}`,
              `≤${plan.max_keyframes} keyframes`,
              `OCR ${plan.ocr_enabled ? "on" : "off"}`,
              `${plan.download_height}p`,
            ].map((t) => (
              <span key={t} className="pill">{t}</span>
            ))}
          </div>
          {(plan.notes || []).map((n: string, i: number) => (
            <p
              key={i}
              className="mt-2 rounded-sm border border-border bg-surface px-3 py-2 text-[0.83rem] text-ink-muted"
            >
              {n}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
