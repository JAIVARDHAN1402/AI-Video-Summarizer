"use client";

import { Check, Clock, MessageSquare, NotebookPen, Pencil, Plus, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, mmss, type Note } from "@/lib/api";
import { Stamp } from "@/components/Bits";

export function Notes({
  videoId,
  sourceRef,
  atSec,
  onSeek,
}: {
  videoId: string;
  sourceRef: string;
  /** The moment the player was last sent to — offered as the note's anchor. */
  atSec: number;
  onSeek: (s: number) => void;
}) {
  const [notes, setNotes] = useState<Note[] | null>(null);
  const [draft, setDraft] = useState("");
  const [anchor, setAnchor] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

  useEffect(() => {
    api.notes(videoId).then(setNotes).catch((e) => setErr(e.message));
  }, [videoId]);

  async function add() {
    const text = draft.trim();
    if (!text || busy) return;
    setBusy(true);
    setErr(null);
    try {
      const n = await api.addNote(videoId, text, anchor && atSec > 0 ? atSec : undefined);
      setNotes((all) => [n, ...(all ?? [])]);
      setDraft("");
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function save(id: string) {
    const text = editText.trim();
    if (!text) return;
    try {
      const n = await api.editNote(videoId, id, text);
      setNotes((all) => (all ?? []).map((x) => (x.id === id ? n : x)));
      setEditing(null);
    } catch (e: any) {
      setErr(e.message);
    }
  }

  async function remove(id: string) {
    // Optimistic: the row disappears immediately and comes back on failure,
    // because a note is small and the round trip is not worth a spinner.
    const before = notes ?? [];
    setNotes(before.filter((n) => n.id !== id));
    try {
      await api.deleteNote(videoId, id);
    } catch (e: any) {
      setErr(e.message);
      setNotes(before);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-[720px] flex-col gap-3 p-6">
          {notes === null ? (
            <>
              <div className="skeleton h-16 w-full" />
              <div className="skeleton h-16 w-full" />
            </>
          ) : notes.length === 0 ? (
            <div className="grid place-items-center py-14 text-center">
              <span className="mb-3 grid h-11 w-11 place-items-center rounded-xl bg-accent/[0.12]">
                <NotebookPen className="h-5 w-5 text-accent" />
              </span>
              <p className="text-[1.05rem] font-bold tracking-tight text-ink">
                Your own notes on this video
              </p>
              <p className="mt-1.5 max-w-[420px] text-[0.87rem] leading-relaxed text-ink-muted">
                Everything else here is generated and can be rebuilt. Notes are yours — they
                survive reprocessing, and each one can point at the moment that prompted it.
              </p>
            </div>
          ) : (
            notes.map((n) => (
              <div key={n.id} className="group rounded-xl border border-border bg-surface-raised p-3.5">
                <div className="mb-1.5 flex items-center gap-2">
                  {n.start_sec !== null && (
                    <Stamp sec={n.start_sec} sourceRef={sourceRef} onSeek={onSeek} />
                  )}
                  {n.source === "chat" && (
                    <span className="pill pill-accent gap-1 !text-[0.62rem]">
                      <MessageSquare className="h-2.5 w-2.5" />
                      from chat
                    </span>
                  )}
                  <span className="ml-auto flex gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                    {editing === n.id ? (
                      <>
                        <button onClick={() => save(n.id)} className="icon-btn !h-7 !w-7" title="Save">
                          <Check className="h-3.5 w-3.5 text-success" />
                        </button>
                        <button
                          onClick={() => setEditing(null)}
                          className="icon-btn !h-7 !w-7"
                          title="Cancel"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={() => {
                            setEditing(n.id);
                            setEditText(n.text);
                          }}
                          className="icon-btn !h-7 !w-7"
                          title="Edit"
                        >
                          <Pencil className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={() => remove(n.id)}
                          className="icon-btn !h-7 !w-7 hover:text-danger"
                          title="Delete"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </>
                    )}
                  </span>
                </div>

                {editing === n.id ? (
                  <textarea
                    autoFocus
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    className="input min-h-[80px] resize-y"
                  />
                ) : (
                  <p className="whitespace-pre-wrap text-[0.88rem] leading-relaxed text-ink">
                    {n.text}
                  </p>
                )}
              </div>
            ))
          )}
          {err && <p className="text-[0.85rem] text-danger">{err}</p>}
        </div>
      </div>

      <div className="flex-shrink-0 border-t border-border p-3">
        <div className="mx-auto max-w-[720px]">
          <div className="composer">
            <textarea
              rows={1}
              value={draft}
              placeholder="Write a note…"
              onChange={(e) => {
                setDraft(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  add();
                }
              }}
            />
            <button
              onClick={add}
              disabled={busy || !draft.trim()}
              aria-label="Add note"
              className="btn btn-primary !h-9 !w-9 !rounded-xl !p-0"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>
          <label className="mt-2 flex cursor-pointer items-center gap-1.5 text-[0.76rem] text-ink-dim">
            <input
              type="checkbox"
              checked={anchor}
              onChange={(e) => setAnchor(e.target.checked)}
              className="accent-[rgb(var(--c-accent))]"
            />
            <Clock className="h-3 w-3" />
            {atSec > 0 ? `Anchor to ${mmss(atSec)}` : "Anchor to the current moment"}
            <span className="ml-auto">Ctrl+Enter to save</span>
          </label>
        </div>
      </div>
    </div>
  );
}
