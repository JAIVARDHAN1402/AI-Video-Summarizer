"use client";

import { ChevronLeft, ChevronRight, RotateCcw, Search, Shuffle } from "lucide-react";
import { useMemo, useState } from "react";
import { mmss } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { Stamp } from "@/components/Bits";

/* ========================================================================== */
/*  Study guide — the explanation re-cut as revision material                  */
/* ========================================================================== */

export function StudyGuide({
  u,
  sourceRef,
  onSeek,
}: {
  u: any;
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const sections: any[] = u?.explanation || [];
  const points: string[] = u?.key_points || [];
  const glossary: any[] = u?.glossary || [];

  return (
    <div className="mx-auto flex max-w-[780px] flex-col gap-7 p-6">
      {u?.overview && (
        <section>
          <h2 className="mb-2 text-[1.05rem] font-bold tracking-tight text-ink">In one paragraph</h2>
          <p className="prose-body text-ink-muted">{u.overview}</p>
        </section>
      )}

      {points.length > 0 && (
        <section>
          <h2 className="mb-3 text-[1.05rem] font-bold tracking-tight text-ink">
            What you should walk away knowing
          </h2>
          <ol className="flex list-none flex-col gap-2 p-0">
            {points.map((p, i) => (
              <li key={i} className="flex gap-3">
                <span className="mt-0.5 grid h-5 w-5 flex-shrink-0 place-items-center rounded-full bg-accent/[0.12] font-mono text-[0.68rem] font-bold text-accent">
                  {i + 1}
                </span>
                <span className="prose-body">{p}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      {sections.length > 0 && (
        <section>
          <h2 className="mb-3 text-[1.05rem] font-bold tracking-tight text-ink">
            Section-by-section recap
          </h2>
          <div className="flex flex-col gap-3">
            {sections.map((s, i) => (
              <details
                key={i}
                className="rounded-lg border border-border bg-surface-raised px-4 py-3"
              >
                <summary className="flex cursor-pointer items-center gap-2.5 font-semibold text-ink">
                  <Stamp sec={s.start_sec ?? 0} sourceRef={sourceRef} onSeek={onSeek} />
                  <span className="min-w-0 flex-1">{s.heading}</span>
                  {s.from_video === false && (
                    <span className="pill pill-accent flex-shrink-0 !text-[0.62rem]">added</span>
                  )}
                </summary>
                <div className="mt-2 border-t border-border pt-2">
                  <Markdown>{s.body || ""}</Markdown>
                </div>
              </details>
            ))}
          </div>
        </section>
      )}

      {glossary.length > 0 && (
        <section>
          <h2 className="mb-3 text-[1.05rem] font-bold tracking-tight text-ink">Terms to know</h2>
          <dl className="m-0 grid gap-x-5 gap-y-3 sm:grid-cols-2">
            {glossary.map((g, i) => (
              <div key={i} className="rounded-lg border border-border bg-surface-raised p-3">
                <dt className="font-semibold text-ink">{g.term}</dt>
                <dd className="m-0 mt-1 text-[0.85rem] leading-relaxed text-ink-muted">
                  {g.definition}
                </dd>
              </div>
            ))}
          </dl>
        </section>
      )}
    </div>
  );
}

/* ========================================================================== */
/*  Flashcards — glossary terms, one at a time, flipped by click               */
/* ========================================================================== */

export function Flashcards({ glossary }: { glossary: any[] }) {
  const [order, setOrder] = useState<number[]>(() => glossary.map((_, i) => i));
  const [at, setAt] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [known, setKnown] = useState<Set<number>>(new Set());

  if (!glossary.length) {
    return <Empty what="flashcards" why="this video produced no glossary terms" />;
  }

  const idx = order[at];
  const card = glossary[idx];

  function go(delta: number) {
    setFlipped(false);
    setAt((a) => Math.min(Math.max(a + delta, 0), order.length - 1));
  }

  function shuffle() {
    const next = [...order];
    for (let i = next.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [next[i], next[j]] = [next[j], next[i]];
    }
    setOrder(next);
    setAt(0);
    setFlipped(false);
  }

  return (
    <div className="mx-auto flex max-w-[620px] flex-col gap-4 p-6">
      <div className="flex items-center justify-between">
        <span className="text-[0.82rem] font-semibold text-ink-muted">
          Card {at + 1} of {order.length}
          {known.size > 0 && (
            <span className="ml-2 text-success">· {known.size} marked known</span>
          )}
        </span>
        <div className="flex gap-1">
          <button onClick={shuffle} className="icon-btn" title="Shuffle">
            <Shuffle className="h-4 w-4" />
          </button>
          <button
            onClick={() => {
              setKnown(new Set());
              setAt(0);
              setFlipped(false);
            }}
            className="icon-btn"
            title="Reset progress"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="h-1.5 overflow-hidden rounded-full bg-surface-active">
        <div
          className="h-full rounded-full bg-accent transition-all"
          style={{ width: `${((at + 1) / order.length) * 100}%` }}
        />
      </div>

      <button
        onClick={() => setFlipped((f) => !f)}
        className="grid min-h-[260px] place-items-center rounded-xl border border-border bg-surface-raised p-8 text-center transition-colors hover:border-border-strong"
      >
        {flipped ? (
          <p className="text-[0.95rem] leading-relaxed text-ink-muted">{card.definition}</p>
        ) : (
          <div>
            <p className="text-[1.4rem] font-bold tracking-tight text-ink">{card.term}</p>
            <p className="mt-3 text-[0.75rem] text-ink-dim">click to reveal</p>
          </div>
        )}
      </button>

      <div className="flex items-center justify-between gap-2">
        <button onClick={() => go(-1)} disabled={at === 0} className="btn !px-3 !py-2">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button
          onClick={() => {
            setKnown((k) => new Set(k).add(idx));
            go(1);
          }}
          className="btn flex-1 text-success"
        >
          I knew this
        </button>
        <button
          onClick={() => go(1)}
          disabled={at === order.length - 1}
          className="btn !px-3 !py-2"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

/* ========================================================================== */
/*  Data table — every scene as a sortable, filterable row                     */
/* ========================================================================== */

type SortKey = "start" | "importance" | "objects";

export function DataTable({
  scenes,
  sourceRef,
  onSeek,
}: {
  scenes: any[];
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortKey>("start");
  const [desc, setDesc] = useState(false);

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    let out = scenes.map((s, i) => ({
      i,
      start: s.start ?? 0,
      end: s.end ?? 0,
      importance: typeof s.importance === "number" ? s.importance : null,
      objects: (s.objects || []).map((o: any) => o.label),
      ocr: s.ocr_text || "",
    }));
    if (needle) {
      out = out.filter(
        (r) =>
          r.ocr.toLowerCase().includes(needle) ||
          r.objects.some((o: string) => o.toLowerCase().includes(needle)),
      );
    }
    out.sort((a, b) => {
      const va =
        sort === "start" ? a.start : sort === "importance" ? (a.importance ?? -1) : a.objects.length;
      const vb =
        sort === "start" ? b.start : sort === "importance" ? (b.importance ?? -1) : b.objects.length;
      return desc ? vb - va : va - vb;
    });
    return out;
  }, [scenes, q, sort, desc]);

  function header(key: SortKey, label: string, extra = "") {
    return (
      <th
        onClick={() => {
          if (sort === key) setDesc((d) => !d);
          else {
            setSort(key);
            setDesc(key !== "start");
          }
        }}
        className={`cursor-pointer select-none border-b border-border px-3 py-2 text-left text-[0.7rem] font-bold uppercase tracking-wide text-ink-dim hover:text-ink ${extra}`}
      >
        {label}
        {sort === key && <span className="ml-1 text-accent">{desc ? "↓" : "↑"}</span>}
      </th>
    );
  }

  if (!scenes.length) return <Empty what="a data table" why="this video has no scene data" />;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-border p-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-dim" />
          <input
            className="input !pl-9"
            placeholder="Filter by on-screen text or detected object…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
        <span className="flex-shrink-0 font-mono text-[0.75rem] tabular-nums text-ink-dim">
          {rows.length}/{scenes.length}
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full border-collapse text-[0.82rem]">
          <thead className="sticky top-0 z-10 bg-surface">
            <tr>
              {header("start", "Time")}
              {header("importance", "Importance")}
              {header("objects", "Objects")}
              <th className="border-b border-border px-3 py-2 text-left text-[0.7rem] font-bold uppercase tracking-wide text-ink-dim">
                On-screen text
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.i} className="border-b border-border hover:bg-surface-hover">
                <td className="whitespace-nowrap px-3 py-2 align-top">
                  <Stamp sec={r.start} sourceRef={sourceRef} onSeek={onSeek} />
                </td>
                <td className="px-3 py-2 align-top">
                  {r.importance === null ? (
                    <span className="text-ink-dim">&mdash;</span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-active">
                        <span
                          className="block h-full rounded-full bg-accent"
                          style={{ width: `${Math.round(r.importance * 100)}%` }}
                        />
                      </span>
                      <span className="font-mono text-[0.72rem] tabular-nums text-ink-dim">
                        {r.importance.toFixed(2)}
                      </span>
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 align-top">
                  <span className="flex flex-wrap gap-1">
                    {r.objects.slice(0, 4).map((o: string, k: number) => (
                      <span
                        key={k}
                        className="rounded-xs border border-border bg-surface-raised px-1.5 py-0.5 text-[0.68rem] text-ink-muted"
                      >
                        {o}
                      </span>
                    ))}
                  </span>
                </td>
                <td className="px-3 py-2 align-top text-ink-muted">
                  <span className="line-clamp-2">{r.ocr || "—"}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function Empty({ what, why }: { what: string; why: string }) {
  return (
    <div className="grid h-full place-items-center p-10 text-center">
      <div>
        <p className="font-semibold text-ink">No {what} for this video</p>
        <p className="mt-1 text-[0.85rem] text-ink-dim">{why}.</p>
      </div>
    </div>
  );
}

export { mmss };
