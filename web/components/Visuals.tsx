"use client";

import { ChevronLeft, ChevronRight, Maximize2, Minus, Plus } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { mmss } from "@/lib/api";
import { Empty } from "@/components/StudioViews";

/* ========================================================================== */
/*  Mind map — the chapter structure as a two-level tree                       */
/* ========================================================================== */

type Node = { label: string; sec: number; children: { label: string; sec: number }[] };

const ROOT_X = 24;
const CH_X = 290;
const LEAF_X = 600;
const ROW = 34;
const LEAF_ROW = 26;

export function MindMap({
  title,
  chapters,
  sections,
  onSeek,
}: {
  title: string;
  chapters: any[];
  sections: any[];
  onSeek: (s: number) => void;
}) {
  const [zoom, setZoom] = useState(1);

  // Each explanation heading is filed under the last chapter that starts
  // before it, so the map mirrors how the video is actually organised.
  const nodes: Node[] = useMemo(() => {
    const chs = [...chapters]
      .map((c) => ({ label: c.title || "Untitled", sec: c.start_sec ?? c.start ?? 0 }))
      .sort((a, b) => a.sec - b.sec);
    if (!chs.length) return [];
    const out: Node[] = chs.map((c) => ({ ...c, children: [] }));
    for (const s of sections) {
      const sec = s.start_sec ?? 0;
      let k = 0;
      for (let i = 0; i < out.length; i++) if (out[i].sec <= sec) k = i;
      out[k].children.push({ label: s.heading || "", sec });
    }
    return out;
  }, [chapters, sections]);

  if (!nodes.length) return <Empty what="a mind map" why="this video produced no chapters" />;

  // Lay rows out so a chapter's block is as tall as its children need.
  let y = 40;
  const placed = nodes.map((n) => {
    const h = Math.max(ROW, n.children.length * LEAF_ROW);
    const row = { node: n, top: y, mid: y + h / 2, h };
    y += h + 14;
    return row;
  });
  const height = y + 30;
  // A centred root reads well for a handful of chapters, but a 30-chapter
  // video pushes it far below the fold with its connectors arriving from
  // nowhere. Past that, anchor it at the top so the map reads as a tree.
  const rootY = height > 640 ? 64 : height / 2;
  const width = LEAF_X + 340;

  const curve = (x1: number, y1: number, x2: number, y2: number) =>
    `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-border p-3">
        <span className="text-[0.8rem] text-ink-dim">
          {nodes.length} chapters &middot; {nodes.reduce((n, c) => n + c.children.length, 0)} sections
        </span>
        <div className="flex items-center gap-1">
          <button onClick={() => setZoom((z) => Math.max(0.5, z - 0.15))} className="icon-btn" title="Zoom out">
            <Minus className="h-4 w-4" />
          </button>
          <span className="w-10 text-center font-mono text-[0.72rem] tabular-nums text-ink-dim">
            {Math.round(zoom * 100)}%
          </span>
          <button onClick={() => setZoom((z) => Math.min(1.8, z + 0.15))} className="icon-btn" title="Zoom in">
            <Plus className="h-4 w-4" />
          </button>
          <button onClick={() => setZoom(1)} className="icon-btn" title="Reset zoom">
            <Maximize2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        <svg
          width={width * zoom}
          height={height * zoom}
          viewBox={`0 0 ${width} ${height}`}
          className="overflow-visible"
          role="img"
          aria-label="Mind map of the video's chapters"
        >
          {placed.map((p, i) => (
            <path
              key={`e${i}`}
              d={curve(ROOT_X + 216, rootY, CH_X, p.mid)}
              fill="none"
              stroke="rgb(var(--c-border-strong))"
              strokeWidth={1.5}
            />
          ))}
          {placed.map((p, i) =>
            p.node.children.map((c, k) => (
              <path
                key={`l${i}-${k}`}
                d={curve(
                  CH_X + 290,
                  p.mid,
                  LEAF_X,
                  p.top + k * LEAF_ROW + LEAF_ROW / 2,
                )}
                fill="none"
                stroke="rgb(var(--c-border))"
                strokeWidth={1.2}
              />
            )),
          )}

          {/* root */}
          <g>
            <rect
              x={ROOT_X}
              y={rootY - 22}
              width={216}
              height={44}
              rx={11}
              fill="rgb(var(--c-accent))"
            />
            <text
              x={ROOT_X + 108}
              y={rootY + 5}
              textAnchor="middle"
              className="fill-white text-[13px] font-bold"
            >
              {clip(title, 26)}
            </text>
          </g>

          {placed.map((p, i) => (
            <g key={`c${i}`}>
              <rect
                x={CH_X}
                y={p.mid - 15}
                width={290}
                height={30}
                rx={8}
                fill="rgb(var(--c-surface-raised))"
                stroke="rgb(var(--c-border-strong))"
              />
              <text
                x={CH_X + 12}
                y={p.mid + 4}
                className="cursor-pointer fill-[rgb(var(--c-ink))] text-[12px] font-semibold"
                onClick={() => onSeek(p.node.sec)}
              >
                {clip(p.node.label, 34)}
              </text>
              <text
                x={CH_X + 280}
                y={p.mid + 4}
                textAnchor="end"
                className="fill-[rgb(var(--c-ink-dim))] font-mono text-[10px]"
              >
                {mmss(p.node.sec)}
              </text>

              {p.node.children.map((c, k) => (
                <text
                  key={k}
                  x={LEAF_X + 8}
                  y={p.top + k * LEAF_ROW + LEAF_ROW / 2 + 4}
                  className="cursor-pointer fill-[rgb(var(--c-ink-muted))] text-[11px]"
                  onClick={() => onSeek(c.sec)}
                >
                  {clip(c.label, 38)}
                </text>
              ))}
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}

function clip(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

/* ========================================================================== */
/*  Slide deck — one slide per chapter, with the nearest keyframe              */
/* ========================================================================== */

export function SlideDeck({
  title,
  chapters,
  scenes,
  onSeek,
}: {
  title: string;
  chapters: any[];
  scenes: any[];
  onSeek: (s: number) => void;
}) {
  const slides = useMemo(() => {
    const chs = [...chapters]
      .map((c) => ({
        title: c.title || "Untitled",
        summary: c.summary || "",
        sec: c.start_sec ?? c.start ?? 0,
      }))
      .sort((a, b) => a.sec - b.sec);
    // Pick the keyframe closest to each chapter's start.
    return chs.map((c) => {
      let best: any = null;
      let bestD = Infinity;
      for (const s of scenes) {
        if (!s.keyframe_url) continue;
        const d = Math.abs((s.start ?? 0) - c.sec);
        if (d < bestD) {
          bestD = d;
          best = s;
        }
      }
      return { ...c, image: best?.keyframe_url as string | undefined };
    });
  }, [chapters, scenes]);

  const [at, setAt] = useState(0);

  useEffect(() => {
    function key(e: KeyboardEvent) {
      if (e.key === "ArrowRight") setAt((a) => Math.min(a + 1, slides.length - 1));
      if (e.key === "ArrowLeft") setAt((a) => Math.max(a - 1, 0));
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [slides.length]);

  if (!slides.length) return <Empty what="a slide deck" why="this video produced no chapters" />;

  const s = slides[Math.min(at, slides.length - 1)];

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1 items-center justify-center p-6">
        <div className="flex aspect-[16/9] w-full max-w-[840px] flex-col overflow-hidden rounded-xl border border-border bg-surface-raised">
          <div className="flex items-start justify-between gap-4 px-8 pt-7">
            <div className="min-w-0">
              <p className="text-[0.7rem] font-bold uppercase tracking-widest text-accent">
                {clip(title, 46)}
              </p>
              <h2 className="mt-1.5 text-[1.5rem] font-bold leading-tight tracking-tight text-ink">
                {s.title}
              </h2>
            </div>
            <button
              onClick={() => onSeek(s.sec)}
              className="ts flex-shrink-0"
              title="Jump to this moment"
            >
              {mmss(s.sec)}
            </button>
          </div>

          <div className="flex min-h-0 flex-1 gap-6 px-8 pb-7 pt-4">
            <p className="prose-body min-w-0 flex-1 overflow-y-auto text-ink-muted">{s.summary}</p>
            {s.image && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={s.image}
                alt=""
                className="h-fit w-[42%] flex-shrink-0 rounded-lg border border-border object-cover"
              />
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-shrink-0 items-center justify-center gap-3 border-t border-border p-3">
        <button onClick={() => setAt((a) => Math.max(a - 1, 0))} disabled={at === 0} className="btn !px-3 !py-2">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <div className="flex max-w-[380px] flex-wrap justify-center gap-1">
          {slides.map((_, i) => (
            <button
              key={i}
              onClick={() => setAt(i)}
              aria-label={`Slide ${i + 1}`}
              className={`h-1.5 rounded-full transition-all ${
                i === at ? "w-6 bg-accent" : "w-1.5 bg-surface-active hover:bg-border-strong"
              }`}
            />
          ))}
        </div>
        <button
          onClick={() => setAt((a) => Math.min(a + 1, slides.length - 1))}
          disabled={at === slides.length - 1}
          className="btn !px-3 !py-2"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
        <span className="ml-2 font-mono text-[0.75rem] tabular-nums text-ink-dim">
          {at + 1}/{slides.length}
        </span>
      </div>
    </div>
  );
}
