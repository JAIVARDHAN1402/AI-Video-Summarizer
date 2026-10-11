"use client";

import {
  AudioLines,
  BookOpen,
  BrainCircuit,
  FileText,
  Image as ImageIcon,
  Layers,
  ListChecks,
  Network,
  Presentation,
  Table2,
} from "lucide-react";
import type { ReactNode } from "react";

export type StudioKind =
  | "study-guide"
  | "flashcards"
  | "mind-map"
  | "data-table"
  | "slide-deck"
  | "quiz"
  | "audio"
  | "infographic"
  | "report";

type Tile = {
  kind: StudioKind;
  label: string;
  icon: ReactNode;
  /** Tailwind classes for the icon chip — one accent per artefact family. */
  tone: string;
  /** What the artefact is derived from, shown as the tile's subtitle. */
  from: string;
  ready: boolean;
};

const TILES: Tile[] = [
  {
    kind: "study-guide",
    label: "Study Guide",
    icon: <BookOpen className="h-[15px] w-[15px]" />,
    tone: "bg-accent/[0.12] text-accent",
    from: "explanation + key points",
    ready: true,
  },
  {
    kind: "flashcards",
    label: "Flashcards",
    icon: <Layers className="h-[15px] w-[15px]" />,
    tone: "bg-violet/[0.12] text-violet",
    from: "glossary terms",
    ready: true,
  },
  {
    kind: "mind-map",
    label: "Mind Map",
    icon: <Network className="h-[15px] w-[15px]" />,
    tone: "bg-cyan/[0.12] text-cyan",
    from: "chapter structure",
    ready: true,
  },
  {
    kind: "data-table",
    label: "Data Table",
    icon: <Table2 className="h-[15px] w-[15px]" />,
    tone: "bg-success/[0.12] text-success",
    from: "scenes + timings",
    ready: true,
  },
  {
    kind: "slide-deck",
    label: "Slide Deck",
    icon: <Presentation className="h-[15px] w-[15px]" />,
    tone: "bg-warning/[0.14] text-warning",
    from: "chapters + keyframes",
    ready: true,
  },
  {
    kind: "report",
    label: "Run Report",
    icon: <FileText className="h-[15px] w-[15px]" />,
    tone: "bg-ink-dim/[0.14] text-ink-muted",
    from: "pipeline timings",
    ready: true,
  },
  {
    kind: "quiz",
    label: "Quiz",
    icon: <ListChecks className="h-[15px] w-[15px]" />,
    tone: "bg-accent/[0.12] text-accent",
    from: "written from the explanation",
    ready: true,
  },
  {
    kind: "audio",
    label: "Audio Overview",
    icon: <AudioLines className="h-[15px] w-[15px]" />,
    tone: "bg-violet/[0.12] text-violet",
    from: "two voices, ~5 minutes",
    ready: true,
  },
  {
    kind: "infographic",
    label: "Infographic",
    icon: <ImageIcon className="h-[15px] w-[15px]" />,
    tone: "bg-cyan/[0.12] text-cyan",
    from: "one-page visual summary",
    ready: true,
  },
];

export function StudioPanel({
  active,
  onOpen,
  counts,
}: {
  active: StudioKind | null;
  onOpen: (kind: StudioKind) => void;
  /** How much source material each artefact has; 0 disables its tile. */
  counts: Partial<Record<StudioKind, number>>;
}) {
  return (
    <div className="flex flex-col gap-4 p-3">
      <div className="rounded-lg border border-accent/20 bg-accent/[0.06] p-3">
        <p className="flex items-center gap-1.5 text-[0.78rem] font-semibold text-accent">
          <BrainCircuit className="h-3.5 w-3.5" />
          Built from this video
        </p>
        <p className="mt-1 text-[0.76rem] leading-relaxed text-ink-muted">
          Every artefact below is derived from the explanation the pipeline already produced —
          nothing is re-generated from scratch.
        </p>
      </div>

      <div className="flex flex-col gap-1.5">
        {TILES.map((t) => {
          const n = counts[t.kind];
          const empty = t.ready && !["quiz", "audio"].includes(t.kind) && n === 0;
          return (
            <button
              key={t.kind}
              onClick={() => onOpen(t.kind)}
              disabled={!t.ready || empty}
              title={
                !t.ready
                  ? "Not available yet"
                  : empty
                    ? `No ${t.label.toLowerCase()} source material in this video`
                    : `Open ${t.label}`
              }
              className={`tile ${active === t.kind ? "border-accent/40 bg-accent/[0.07]" : ""}`}
            >
              <span className={`tile-icon ${t.tone}`}>{t.icon}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate">{t.label}</span>
                <span className="mt-0.5 block truncate text-[0.7rem] font-normal text-ink-dim">
                  {t.from}
                </span>
              </span>
              {!t.ready && (
                <span className="pill pill-accent flex-shrink-0 !px-2 !py-0.5 !text-[0.62rem]">
                  Soon
                </span>
              )}
              {t.ready && typeof n === "number" && n > 0 && (
                <span className="flex-shrink-0 font-mono text-[0.72rem] tabular-nums text-ink-dim">
                  {n}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
