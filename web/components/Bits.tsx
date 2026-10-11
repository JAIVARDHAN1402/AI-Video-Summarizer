"use client";

import { mmss, ytId } from "@/lib/api";
import { useState } from "react";

export function SectionHead({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="section-head">
      <h2>{title}</h2>
      {hint && <span>{hint}</span>}
    </div>
  );
}

export function Stamp({
  sec,
  sourceRef,
  onSeek,
}: {
  sec: number;
  sourceRef?: string;
  onSeek?: (s: number) => void;
}) {
  const id = sourceRef ? ytId(sourceRef) : null;
  if (onSeek) {
    return (
      <button className="ts" onClick={() => onSeek(sec)}>
        {mmss(sec)}
      </button>
    );
  }
  if (id) {
    return (
      <a
        className="ts"
        href={`https://youtu.be/${id}?t=${Math.floor(sec)}s`}
        target="_blank"
        rel="noreferrer"
      >
        {mmss(sec)}
      </a>
    );
  }
  return <span className="ts">{mmss(sec)}</span>;
}

export function VideoPlayer({
  sourceRef,
  startSec,
  autoPlay = false,
}: {
  sourceRef: string;
  startSec: number;
  /** Set when the reader asked for this exact moment, e.g. by clicking a
   *  scene. The embed is remounted anyway on every seek, so this only ever
   *  starts playback the reader just requested. */
  autoPlay?: boolean;
}) {
  const id = ytId(sourceRef);
  if (!id) {
    return (
      <div className="grid aspect-video place-items-center rounded border border-border bg-surface px-4 text-center text-[0.85rem] text-ink-dim">
        This source has no embeddable player, so scenes cannot be played back here.
      </div>
    );
  }
  return (
    <div className="relative aspect-video overflow-hidden rounded border border-border bg-black">
      <iframe
        key={`${startSec}-${autoPlay}`}
        className="absolute inset-0 h-full w-full"
        src={
          `https://www.youtube.com/embed/${id}?start=${Math.floor(startSec)}&rel=0` +
          (autoPlay ? "&autoplay=1" : "")
        }
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; picture-in-picture"
        allowFullScreen
      />
    </div>
  );
}

export function CodeTabs({ impls }: { impls: { language: string; code: string }[] }) {
  const [active, setActive] = useState(0);
  if (!impls.length) return null;
  return (
    <div>
      <div className="mb-2 flex w-fit gap-0.5 rounded-sm border border-border bg-bg-elevated p-1">
        {impls.map((im, i) => (
          <button
            key={im.language + i}
            onClick={() => setActive(i)}
            className={`rounded-xs px-3 py-1.5 text-[0.8rem] font-semibold transition-colors ${
              i === active
                ? "bg-surface text-ink shadow-card"
                : "text-ink-dim hover:text-ink"
            }`}
          >
            {im.language}
          </button>
        ))}
      </div>
      <pre className="max-h-[520px] overflow-auto rounded border border-transparent bg-code-bg p-4 font-mono text-[0.82rem] leading-relaxed text-code-ink">
        <code>{impls[active].code}</code>
      </pre>
    </div>
  );
}
