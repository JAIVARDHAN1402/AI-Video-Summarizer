"use client";

import {
  ArrowLeft,
  BookOpen,
  Clapperboard,
  FileAudio,
  MessageSquare,
  NotebookPen,
  Play,
  Sparkles,
  Video,
} from "lucide-react";
import { use, useEffect, useMemo, useState } from "react";
import { api, humanDuration, mmss, type VideoResult } from "@/lib/api";
import { SectionHead, Stamp, VideoPlayer } from "@/components/Bits";
import { UnderstandingView } from "@/components/UnderstandingView";
import { Workspace } from "@/components/Workspace";
import { StudioPanel, type StudioKind } from "@/components/StudioPanel";
import { Chat } from "@/components/Chat";
import { DataTable, Empty, Flashcards, StudyGuide } from "@/components/StudioViews";
import { MindMap, SlideDeck } from "@/components/Visuals";
import { Quiz } from "@/components/Quiz";
import { Notes } from "@/components/Notes";
import { Infographic } from "@/components/Infographic";
import { AudioOverview } from "@/components/AudioOverview";

const TABS = ["Explanation", "Scenes", "Transcript", "Chat", "Notes"] as const;
type Tab = (typeof TABS)[number];

const TAB_ICON: Record<Tab, React.ReactNode> = {
  Explanation: <BookOpen className="h-[15px] w-[15px]" />,
  Scenes: <Clapperboard className="h-[15px] w-[15px]" />,
  Transcript: <FileAudio className="h-[15px] w-[15px]" />,
  Chat: <MessageSquare className="h-[15px] w-[15px]" />,
  Notes: <NotebookPen className="h-[15px] w-[15px]" />,
};

const STUDIO_TITLE: Record<StudioKind, string> = {
  "study-guide": "Study Guide",
  flashcards: "Flashcards",
  "mind-map": "Mind Map",
  "data-table": "Data Table",
  "slide-deck": "Slide Deck",
  report: "Run Report",
  quiz: "Quiz",
  audio: "Audio Overview",
  infographic: "Infographic",
};

export default function VideoPage({ params }: { params: Promise<{ videoId: string }> }) {
  const { videoId } = use(params);
  const [data, setData] = useState<VideoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Explanation");
  const [studio, setStudio] = useState<StudioKind | null>(null);
  const [seek, setSeek] = useState(0);

  useEffect(() => {
    api.video(videoId).then(setData).catch((e) => setError(e.message));
  }, [videoId]);

  const sourceRef: string = data?.meta?.source_ref || "";
  // Seeking always returns to the player, which lives in the left panel —
  // so unlike the old single-column page there is nothing to scroll.
  const onSeek = (s: number) => setSeek(s);

  const u = data?.understanding;
  const scenes = data?.scenes?.scenes || [];
  const transcript = data?.transcript?.segments || [];
  const timings = (data?.state?.timings || {}) as Record<string, number>;
  const dur = data?.meta?.duration_sec || 0;
  const chapters = u?.chapters || data?.summary?.chapters || [];

  const counts = useMemo(
    () => ({
      "study-guide": (u?.explanation?.length || 0) + (u?.key_points?.length || 0),
      flashcards: u?.glossary?.length || 0,
      "mind-map": chapters.length,
      "data-table": scenes.length,
      "slide-deck": chapters.length,
      infographic: u ? 1 : 0,
      report: Object.keys(timings).length,
    }),
    [u, chapters, scenes, timings],
  );

  if (error)
    return (
      <div className="page-col">
        <div className="card p-6 text-[0.9rem] text-danger">{error}</div>
      </div>
    );

  if (!data)
    return (
      <div className="workspace">
        <div className="panel w-[clamp(240px,22vw,340px)] flex-shrink-0 p-3">
          <div className="skeleton aspect-video w-full" />
          <div className="skeleton mt-3 h-4 w-3/4" />
          <div className="skeleton mt-2 h-4 w-1/2" />
        </div>
        <div className="panel flex-1 p-6">
          <div className="skeleton h-5 w-1/3" />
          <div className="skeleton mt-4 h-3 w-full" />
          <div className="skeleton mt-2 h-3 w-5/6" />
          <div className="skeleton mt-2 h-3 w-4/6" />
        </div>
      </div>
    );

  return (
    <Workspace
      left={{
        title: "Source",
        icon: <Video className="h-[15px] w-[15px]" />,
        children: (
          <SourcePanel
            data={data}
            sourceRef={sourceRef}
            seek={seek}
            chapters={chapters}
            onSeek={onSeek}
          />
        ),
      }}
      right={{
        title: "Studio",
        icon: <Sparkles className="h-[15px] w-[15px]" />,
        children: (
          <StudioPanel
            active={studio}
            counts={counts}
            onOpen={(k) => setStudio((cur) => (cur === k ? null : k))}
          />
        ),
      }}
    >
      {/* ---- canvas header: tabs, or the open artefact's title ---- */}
      <div className="panel-head">
        {studio ? (
          <>
            <button
              onClick={() => setStudio(null)}
              className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[0.82rem] font-semibold text-ink-muted transition-colors hover:bg-surface-hover hover:text-ink"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </button>
            <span className="panel-title">{STUDIO_TITLE[studio]}</span>
            <span className="w-16" />
          </>
        ) : (
          <div className="flex gap-0.5 overflow-x-auto">
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-[0.82rem] font-semibold transition-colors ${
                  tab === t
                    ? "bg-accent/[0.10] text-accent"
                    : "text-ink-dim hover:bg-surface-hover hover:text-ink"
                }`}
              >
                {TAB_ICON[t]}
                {t}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ---- canvas body ---- */}
      {studio ? (
        <StudioCanvas
          kind={studio}
          data={data}
          u={u}
          chapters={chapters}
          scenes={scenes}
          timings={timings}
          dur={dur}
          sourceRef={sourceRef}
          onSeek={onSeek}
        />
      ) : tab === "Notes" ? (
        <Notes videoId={videoId} sourceRef={sourceRef} atSec={seek} onSeek={onSeek} />
      ) : tab === "Chat" ? (
        <Chat videoId={videoId} sourceRef={sourceRef} onSeek={onSeek} />
      ) : tab === "Transcript" ? (
        <TranscriptView segments={transcript} sourceRef={sourceRef} onSeek={onSeek} />
      ) : tab === "Scenes" ? (
        <div className="panel-body p-5">
          <ScenesView
            scenes={scenes}
            highlight={data.highlight}
            sourceRef={sourceRef}
            onSeek={onSeek}
          />
        </div>
      ) : (
        <div className="panel-body">
          <div className="mx-auto max-w-[820px] p-6">
            {u ? (
              <UnderstandingView u={u} sourceRef={sourceRef} onSeek={onSeek} />
            ) : (
              <FallbackSummary summary={data.summary} sourceRef={sourceRef} onSeek={onSeek} />
            )}
          </div>
        </div>
      )}
    </Workspace>
  );
}

/* ========================================================================== */
/*  Left panel — the video itself, its vital statistics, and its chapters      */
/* ========================================================================== */

function SourcePanel({
  data,
  sourceRef,
  seek,
  chapters,
  onSeek,
}: {
  data: VideoResult;
  sourceRef: string;
  seek: number;
  chapters: any[];
  onSeek: (s: number) => void;
}) {
  const dur = data.meta?.duration_sec || 0;
  const timings = (data.state?.timings || {}) as Record<string, number>;
  const words = (data.transcript?.segments || []).reduce(
    (n, s) => n + s.text.split(/\s+/).length,
    0,
  );

  return (
    <div className="flex flex-col">
      <div className="p-3">
        <VideoPlayer sourceRef={sourceRef} startSec={seek} />
        <h1 className="mt-3 text-[0.95rem] font-bold leading-snug tracking-tight text-ink">
          {data.meta?.title || "Untitled video"}
        </h1>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {dur > 0 && <span className="pill">{humanDuration(dur)}</span>}
          {timings.total && (
            <span className="pill pill-success">in {humanDuration(timings.total)}</span>
          )}
          {timings.total && dur > 0 && (
            <span className="pill pill-accent">{Math.round(dur / timings.total)}&times; faster</span>
          )}
        </div>

        <dl className="mt-3 grid grid-cols-3 gap-1.5">
          {[
            [String(chapters.length), "Chapters"],
            [String((data.scenes?.scenes || []).length), "Scenes"],
            [words > 999 ? `${Math.round(words / 1000)}k` : String(words), "Words"],
          ].map(([v, l]) => (
            <div key={l} className="rounded-md border border-border bg-surface-raised px-2 py-1.5">
              <dt className="text-[0.6rem] font-bold uppercase tracking-wide text-ink-dim">{l}</dt>
              <dd className="m-0 text-[0.95rem] font-bold tabular-nums text-ink">{v}</dd>
            </div>
          ))}
        </dl>
      </div>

      {chapters.length > 0 && (
        <div className="border-t border-border px-3 py-2.5">
          <p className="mb-1.5 text-[0.68rem] font-bold uppercase tracking-wide text-ink-dim">
            Chapters
          </p>
          <div className="flex flex-col">
            {chapters.map((c: any, i: number) => (
              <button
                key={i}
                onClick={() => onSeek(c.start_sec ?? c.start ?? 0)}
                className="flex gap-2 rounded-md px-1.5 py-1.5 text-left transition-colors hover:bg-surface-hover"
              >
                <span className="ts flex-shrink-0 !text-[0.68rem]">
                  {Math.floor((c.start_sec ?? c.start ?? 0) / 60)}:
                  {String(Math.floor((c.start_sec ?? c.start ?? 0) % 60)).padStart(2, "0")}
                </span>
                <span className="min-w-0 text-[0.79rem] font-medium leading-snug text-ink-muted">
                  {c.title}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ========================================================================== */
/*  Studio canvas — routes a tile to the artefact it generates                 */
/* ========================================================================== */

function StudioCanvas({
  kind,
  data,
  u,
  chapters,
  scenes,
  timings,
  dur,
  sourceRef,
  onSeek,
}: {
  kind: StudioKind;
  data: VideoResult;
  u: any;
  chapters: any[];
  scenes: any[];
  timings: Record<string, number>;
  dur: number;
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const title = data.meta?.title || "This video";

  switch (kind) {
    case "study-guide":
      return (
        <div className="panel-body">
          <StudyGuide u={u} sourceRef={sourceRef} onSeek={onSeek} />
        </div>
      );
    case "flashcards":
      return (
        <div className="panel-body">
          <Flashcards glossary={u?.glossary || []} />
        </div>
      );
    case "mind-map":
      return (
        <MindMap
          title={title}
          chapters={chapters}
          sections={u?.explanation || []}
          onSeek={onSeek}
        />
      );
    case "data-table":
      return <DataTable scenes={scenes} sourceRef={sourceRef} onSeek={onSeek} />;
    case "slide-deck":
      return <SlideDeck title={title} chapters={chapters} scenes={scenes} onSeek={onSeek} />;
    case "infographic":
      return (
        <Infographic
          title={title}
          u={u}
          scenes={scenes}
          durationSec={dur}
          timings={timings}
          onSeek={onSeek}
        />
      );
    case "audio":
      return <AudioOverview videoId={data.video_id} hasUnderstanding={!!u} />;
    case "quiz":
      return (
        <Quiz
          videoId={data.video_id}
          hasUnderstanding={!!u}
          sourceRef={sourceRef}
          onSeek={onSeek}
        />
      );
    case "report":
      return (
        <div className="panel-body p-6">
          <ReportView timings={timings} dur={dur} meta={data.meta} />
        </div>
      );
    default:
      return <Empty what="that artefact" why="it is not available yet" />;
  }
}

/* ========================================================================== */
/*  Views carried over from the single-column page                            */
/* ========================================================================== */

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
          <SectionHead
            title="Walkthrough"
            hint="Local model — set GEMINI_API_KEY for the full explanation"
          />
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
            className="w-full rounded-lg border border-border bg-black"
          />
        </section>
      )}

      {hasImportance && (
        <div className="flex items-center gap-2 text-[0.8rem]">
          <span className="text-ink-dim">Order</span>
          <div className="flex gap-0.5 rounded-sm border border-border bg-bg-elevated p-1">
            {[
              ["Timeline", false],
              ["Importance", true],
            ].map(([label, val]) => (
              <button
                key={label as string}
                onClick={() => setByImportance(val as boolean)}
                className={`rounded-xs px-2.5 py-1 font-semibold transition-colors ${
                  byImportance === val
                    ? "bg-surface text-ink shadow-card"
                    : "text-ink-dim hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3.5 sm:grid-cols-3 xl:grid-cols-4">
        {ordered.map((s: any, i: number) => (
          <div
            key={i}
            // The whole card seeks, not just its timestamp: a grid of frames
            // reads as something you click, and only the small stamp being
            // live made it look broken.
            role="button"
            tabIndex={0}
            onClick={() => onSeek(s.start)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSeek(s.start);
              }
            }}
            title={`Play from ${mmss(s.start)}`}
            className="group/scene cursor-pointer overflow-hidden rounded-lg border border-border bg-surface pb-2 transition-all hover:border-accent/50 hover:shadow-card-hover focus:outline-none focus-visible:border-accent"
          >
            {s.keyframe_url && (
              <div className="relative">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={s.keyframe_url}
                  alt=""
                  loading="lazy"
                  className="aspect-video w-full object-cover"
                />
                <span className="absolute inset-0 grid place-items-center bg-black/35 opacity-0 transition-opacity group-hover/scene:opacity-100">
                  <span className="grid h-9 w-9 place-items-center rounded-full bg-white/90">
                    <Play className="ml-0.5 h-4 w-4 text-black" fill="currentColor" />
                  </span>
                </span>
              </div>
            )}
            {typeof s.importance === "number" && (
              <div className="h-1 w-full bg-surface-active">
                <div
                  className="h-full bg-accent"
                  style={{ width: `${Math.round(s.importance * 100)}%` }}
                  title={`importance ${(s.importance * 100).toFixed(0)}%`}
                />
              </div>
            )}
            <div
              className="mt-2 flex flex-wrap items-center gap-1.5 px-2.5"
              onClick={(e) => e.stopPropagation()}
            >
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
      q.trim() ? segments.filter((s) => s.text.toLowerCase().includes(q.toLowerCase())) : segments,
    [q, segments],
  );
  return (
    <div className="flex h-full flex-col">
      <div className="flex-shrink-0 border-b border-border p-3">
        <input
          className="input"
          placeholder="Search the transcript…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <div className="mx-auto max-w-[760px]">
          {filtered.slice(0, 800).map((s, i) => (
            <div key={i} className="flex gap-3 rounded-md px-2 py-1.5 hover:bg-surface-hover">
              <Stamp sec={s.start} sourceRef={sourceRef} onSeek={onSeek} />
              <span className="text-[0.885rem] leading-relaxed text-ink">{s.text}</span>
            </div>
          ))}
        </div>
      </div>
      <p className="flex-shrink-0 border-t border-border px-3 py-2 text-[0.76rem] text-ink-dim">
        {filtered.length.toLocaleString()} of {segments.length.toLocaleString()} lines
        {filtered.length > 800 && " — showing the first 800"}
      </p>
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
  const longest = Math.max(
    ...Object.entries(timings)
      .filter(([k]) => k !== "total")
      .map(([, v]) => v),
    1,
  );
  return (
    <div className="mx-auto flex max-w-[780px] flex-col gap-6">
      {total && dur > 0 && (
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          {[
            [humanDuration(total), "Total time"],
            [`${Math.round(dur / total)}×`, "Faster than watching"],
            [(total / dur).toFixed(2), "Sec per video-sec"],
            [humanDuration(dur), "Video length"],
          ].map(([v, l]) => (
            <div key={l} className="rounded-lg border border-border bg-surface-raised p-3.5">
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
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-active">
                <div
                  className="h-full rounded-full bg-accent"
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
              <span key={t} className="pill">
                {t}
              </span>
            ))}
          </div>
          {(plan.notes || []).map((n: string, i: number) => (
            <p
              key={i}
              className="mt-2 rounded-md border border-border bg-surface px-3 py-2 text-[0.83rem] text-ink-muted"
            >
              {n}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
