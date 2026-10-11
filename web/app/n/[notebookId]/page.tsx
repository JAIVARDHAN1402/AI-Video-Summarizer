"use client";

import {
  AlertTriangle,
  Check,
  Layers,
  MessageSquare,
  Pencil,
  Plus,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import Link from "next/link";
import { use, useEffect, useMemo, useState } from "react";
import { api, humanDuration, type Notebook, type VideoSummary } from "@/lib/api";
import { Chat } from "@/components/Chat";
import { Workspace } from "@/components/Workspace";

export default function NotebookPage({
  params,
}: {
  params: Promise<{ notebookId: string }>;
}) {
  const { notebookId } = use(params);
  const [nb, setNb] = useState<Notebook | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.notebook(notebookId).then(setNb).catch((e) => setErr(e.message));
  }, [notebookId]);

  async function setVideos(ids: string[]) {
    const updated = await api.updateNotebook(notebookId, { video_ids: ids });
    setNb(updated);
  }

  if (err)
    return (
      <div className="page-col">
        <div className="card p-6 text-[0.9rem] text-danger">{err}</div>
      </div>
    );

  if (!nb)
    return (
      <div className="workspace">
        <div className="panel w-[clamp(240px,22vw,340px)] flex-shrink-0 p-3">
          <div className="skeleton h-5 w-2/3" />
          <div className="skeleton mt-3 h-12 w-full" />
          <div className="skeleton mt-2 h-12 w-full" />
        </div>
        <div className="panel flex-1 p-6">
          <div className="skeleton h-5 w-1/3" />
        </div>
      </div>
    );

  const live = nb.videos.filter((v) => !v.missing);

  return (
    <Workspace
      left={{
        title: "Sources",
        icon: <Layers className="h-[15px] w-[15px]" />,
        children: <SourcesPanel nb={nb} onChange={setVideos} />,
      }}
      right={{
        title: "About",
        icon: <Sparkles className="h-[15px] w-[15px]" />,
        children: <AboutPanel nb={nb} onRename={async (t) => setNb(await api.updateNotebook(notebookId, { title: t }))} />,
      }}
    >
      <div className="panel-head">
        <span className="panel-title">
          <MessageSquare className="h-[15px] w-[15px] text-ink-dim" />
          {nb.title}
        </span>
        <span className="text-[0.76rem] text-ink-dim">
          {live.length} source{live.length === 1 ? "" : "s"}
        </span>
      </div>

      {live.length === 0 ? (
        <div className="grid h-full place-items-center p-10 text-center">
          <div className="max-w-[420px]">
            <p className="font-semibold text-ink">No sources in this notebook yet</p>
            <p className="mt-1.5 text-[0.86rem] leading-relaxed text-ink-muted">
              Add processed videos from the Sources panel. Questions are then answered across
              all of them at once, with each answer saying which video it came from.
            </p>
          </div>
        </div>
      ) : (
        <Chat
          videoId={live[0].video_id}
          notebookId={nb.id}
          sourceRef=""
          onSeek={() => undefined}
          starters={[
            "What do these sources agree on?",
            "Where do they disagree?",
            "Which one explains it best, and why?",
            "Give me a combined summary",
          ]}
          intro={{
            title: `Ask across ${live.length} videos`,
            body:
              "Every question is answered using all the sources at once. Answers name which " +
              "video each point came from, say when one covers something better than another, " +
              "and say plainly when two of them disagree.",
          }}
        />
      )}
    </Workspace>
  );
}

/* ========================================================================== */

function SourcesPanel({
  nb,
  onChange,
}: {
  nb: Notebook;
  onChange: (ids: string[]) => Promise<void>;
}) {
  const [adding, setAdding] = useState(false);
  const [all, setAll] = useState<VideoSummary[] | null>(null);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (adding && all === null) api.videos().then(setAll).catch(() => setAll([]));
  }, [adding, all]);

  const candidates = useMemo(() => {
    const have = new Set(nb.video_ids);
    const needle = q.trim().toLowerCase();
    return (all ?? [])
      .filter((v) => !have.has(v.video_id))
      .filter((v) => (needle ? v.title.toLowerCase().includes(needle) : true));
  }, [all, nb.video_ids, q]);

  async function add(id: string) {
    setBusy(true);
    try {
      await onChange([...nb.video_ids, id]);
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    setBusy(true);
    try {
      await onChange(nb.video_ids.filter((v) => v !== id));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col">
      <div className="flex flex-col gap-1.5 p-3">
        {nb.videos.map((v) => (
          <div
            key={v.video_id}
            className={`group flex items-start gap-2 rounded-lg border p-2.5 ${
              v.missing ? "border-warning/30 bg-warning/[0.05]" : "border-border bg-surface-raised"
            }`}
          >
            <span className="mt-0.5 flex-shrink-0">
              {v.missing ? (
                <AlertTriangle className="h-3.5 w-3.5 text-warning" />
              ) : (
                <Layers className="h-3.5 w-3.5 text-ink-dim" />
              )}
            </span>
            <div className="min-w-0 flex-1">
              {v.missing ? (
                <p className="text-[0.78rem] leading-snug text-warning">
                  No longer in the cache
                </p>
              ) : (
                <Link
                  href={`/v/${v.video_id}`}
                  className="block truncate text-[0.79rem] font-medium leading-snug text-ink no-underline hover:text-accent"
                  title={v.title}
                >
                  {v.title}
                </Link>
              )}
              {v.duration_sec ? (
                <p className="mt-0.5 text-[0.68rem] text-ink-dim">
                  {humanDuration(v.duration_sec)}
                </p>
              ) : null}
            </div>
            <button
              onClick={() => remove(v.video_id)}
              disabled={busy}
              className="icon-btn !h-6 !w-6 flex-shrink-0 opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
              title="Remove from this notebook"
              aria-label="Remove from this notebook"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ))}

        <button
          onClick={() => setAdding((a) => !a)}
          className="tile mt-1 justify-center !py-2 !text-[0.78rem]"
        >
          {adding ? <X className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
          {adding ? "Done adding" : "Add a video"}
        </button>
      </div>

      {adding && (
        <div className="border-t border-border p-3">
          <div className="relative mb-2">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-dim" />
            <input
              className="input !py-1.5 !pl-8 !text-[0.8rem]"
              placeholder="Search processed videos…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          {all === null ? (
            <div className="skeleton h-8 w-full" />
          ) : candidates.length === 0 ? (
            <p className="py-3 text-center text-[0.78rem] text-ink-dim">
              {nb.video_ids.length > 0 ? "Everything else is already here" : "Nothing processed yet"}
            </p>
          ) : (
            <div className="flex max-h-[320px] flex-col gap-1 overflow-y-auto">
              {candidates.slice(0, 40).map((v) => (
                <button
                  key={v.video_id}
                  onClick={() => add(v.video_id)}
                  disabled={busy}
                  className="flex items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors hover:bg-surface-hover"
                >
                  <Plus className="h-3 w-3 flex-shrink-0 text-accent" />
                  <span className="min-w-0 flex-1 truncate text-[0.78rem] text-ink">
                    {v.title}
                  </span>
                  {v.has_understanding && (
                    <Sparkles className="h-3 w-3 flex-shrink-0 text-accent" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AboutPanel({
  nb,
  onRename,
}: {
  nb: Notebook;
  onRename: (title: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(nb.title);
  const live = nb.videos.filter((v) => !v.missing);
  const totalSec = live.reduce((n, v) => n + (v.duration_sec || 0), 0);

  return (
    <div className="flex flex-col gap-4 p-3">
      <div>
        <p className="mb-1.5 text-[0.68rem] font-bold uppercase tracking-wide text-ink-dim">
          Notebook
        </p>
        {editing ? (
          <div className="flex gap-1">
            <input
              autoFocus
              className="input !py-1.5 !text-[0.85rem]"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  onRename(title);
                  setEditing(false);
                }
              }}
            />
            <button
              onClick={() => {
                onRename(title);
                setEditing(false);
              }}
              className="icon-btn"
              aria-label="Save title"
            >
              <Check className="h-4 w-4 text-success" />
            </button>
          </div>
        ) : (
          <div className="group flex items-start gap-1.5">
            <p className="min-w-0 flex-1 font-semibold leading-snug text-ink">{nb.title}</p>
            <button
              onClick={() => setEditing(true)}
              className="icon-btn !h-6 !w-6 flex-shrink-0 opacity-0 group-hover:opacity-100"
              aria-label="Rename"
            >
              <Pencil className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>

      <dl className="grid grid-cols-2 gap-1.5">
        {[
          [String(live.length), "Sources"],
          [totalSec ? humanDuration(totalSec) : "—", "Total runtime"],
        ].map(([v, l]) => (
          <div key={l} className="rounded-md border border-border bg-surface-raised px-2 py-1.5">
            <dt className="text-[0.6rem] font-bold uppercase tracking-wide text-ink-dim">{l}</dt>
            <dd className="m-0 text-[0.92rem] font-bold tabular-nums text-ink">{v}</dd>
          </div>
        ))}
      </dl>

      <div className="rounded-lg border border-accent/20 bg-accent/[0.06] p-3">
        <p className="text-[0.78rem] font-semibold text-accent">Answered across all of them</p>
        <p className="mt-1 text-[0.76rem] leading-relaxed text-ink-muted">
          Each question searches every source&apos;s index at once and ranks the passages
          together, so the answer can say which video covers something best &mdash; and when
          two of them disagree.
        </p>
      </div>

      <Link href="/notebooks" className="btn justify-center !py-1.5 !text-[0.78rem] no-underline">
        All notebooks
      </Link>
    </div>
  );
}
