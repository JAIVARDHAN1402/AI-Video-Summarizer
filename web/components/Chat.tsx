"use client";

import {
  ArrowUp,
  BookmarkPlus,
  Check,
  History,
  MessageSquarePlus,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ChatThread } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { Stamp } from "@/components/Bits";

export type Turn = {
  q: string;
  a: string;
  confidence: string;
  sources: {
    start: number;
    end: number;
    text: string;
    video_id?: string;
    video_title?: string;
  }[];
  beyondVideo?: boolean;
  offTopic?: boolean;
  error?: boolean;
};

const STARTERS = [
  "Explain the core idea in simple terms",
  "What are the key takeaways?",
  "Where does it get difficult?",
  "Give me an example",
];

export function Chat({
  videoId,
  notebookId,
  sourceRef,
  onSeek,
  starters = STARTERS,
  intro,
}: {
  /** The video a single-source chat is about; also where notes are saved. */
  videoId: string;
  /** When set, questions go to the whole notebook instead of one video. */
  notebookId?: string;
  sourceRef: string;
  onSeek: (s: number) => void;
  starters?: string[];
  intro?: { title: string; body: string };
}) {
  const scope = notebookId ? `notebook:${notebookId}` : `video:${videoId}`;

  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  // Shown as a bubble while the answer is in flight — clicking a starter and
  // watching your own question vanish for ten seconds reads as a dropped click.
  const [pending, setPending] = useState<string | null>(null);
  const [history, setHistory] = useState<Turn[]>([]);
  const [saved, setSaved] = useState<Set<number>>(new Set());
  const [threadId, setThreadId] = useState<string | null>(null);
  const [threads, setThreads] = useState<ChatThread[] | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLTextAreaElement>(null);

  const refreshThreads = useCallback(() => {
    api.chats(scope).then(setThreads).catch(() => setThreads([]));
  }, [scope]);

  useEffect(() => {
    refreshThreads();
  }, [refreshThreads]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [history, busy]);

  /** Persist after every exchange — a reload or a tab switch should not cost
   *  a conversation. The first question names the thread. */
  async function persist(turns: Turn[]) {
    try {
      const title = turns[0]?.q.slice(0, 80) || "New chat";
      if (threadId) {
        await api.saveChat(threadId, { turns, title });
      } else {
        const t = await api.createChat(scope, title, turns);
        setThreadId(t.id);
      }
      refreshThreads();
    } catch {
      /* the conversation is still on screen; losing the save is not fatal */
    }
  }

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    setQ("");
    setPending(question);
    setBusy(true);
    try {
      const prior = history.filter((t) => !t.error).map(({ q, a }) => ({ q, a }));
      const r = notebookId
        ? await api.askNotebook(notebookId, question, prior)
        : await api.ask(videoId, question, prior);
      const next: Turn[] = [
        ...history,
        {
          q: question,
          a: r.answer,
          confidence: r.confidence,
          sources: r.sources,
          beyondVideo: r.beyond_video,
          offTopic: r.off_topic,
        },
      ];
      setHistory(next);
      persist(next);
    } catch (err: any) {
      setHistory((h) => [
        ...h,
        { q: question, a: `Error: ${err.message}`, confidence: "low", sources: [], error: true },
      ]);
    } finally {
      setPending(null);
      setBusy(false);
      boxRef.current?.focus();
    }
  }

  async function keep(i: number, t: Turn) {
    try {
      await api.addNote(videoId, `**${t.q}**\n\n${t.a}`, t.sources?.[0]?.start, "chat");
      setSaved((s) => new Set(s).add(i));
    } catch {
      /* the Notes tab is the source of truth; a failed save is visible there */
    }
  }

  function newChat() {
    setHistory([]);
    setThreadId(null);
    setSaved(new Set());
    setShowHistory(false);
    boxRef.current?.focus();
  }

  async function open(id: string) {
    try {
      const t = await api.chat(id);
      setHistory((t.turns || []) as Turn[]);
      setThreadId(t.id);
      setSaved(new Set());
      setShowHistory(false);
    } catch {
      /* ignore — the list refreshes on the next save */
    }
  }

  async function remove(id: string) {
    const before = threads ?? [];
    setThreads(before.filter((t) => t.id !== id));
    if (id === threadId) newChat();
    try {
      await api.deleteChat(id);
    } catch {
      setThreads(before);
    }
  }

  return (
    <div className="relative flex h-full flex-col">
      {/* ---- thread bar ---- */}
      <div className="flex flex-shrink-0 items-center gap-1 border-b border-border px-2 py-1.5">
        <button
          onClick={() => setShowHistory((v) => !v)}
          className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-[0.78rem] font-medium transition-colors ${
            showHistory
              ? "bg-accent/[0.12] text-accent"
              : "text-ink-dim hover:bg-surface-hover hover:text-ink"
          }`}
          title="Past conversations"
        >
          <History className="h-3.5 w-3.5" />
          History
          {threads && threads.length > 0 && (
            <span className="font-mono text-[0.7rem] tabular-nums">{threads.length}</span>
          )}
        </button>

        <span className="min-w-0 flex-1 truncate px-1 text-center text-[0.78rem] text-ink-dim">
          {threadId ? threads?.find((t) => t.id === threadId)?.title : "New chat"}
        </span>

        <button
          onClick={newChat}
          disabled={history.length === 0}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[0.78rem] font-medium text-ink-dim transition-colors hover:bg-surface-hover hover:text-ink disabled:opacity-40"
          title="Start a new conversation"
        >
          <MessageSquarePlus className="h-3.5 w-3.5" />
          New
        </button>
      </div>

      {/* ---- history drawer ---- */}
      {showHistory && (
        <>
          <button
            onClick={() => setShowHistory(false)}
            aria-label="Close history"
            className="absolute inset-0 z-20 bg-black/25"
          />
          <div
            className="absolute inset-y-0 left-0 z-30 flex w-[min(320px,80%)] flex-col border-r border-border bg-surface"
            style={{ boxShadow: "var(--shadow-pop)" }}
          >
            <div className="flex h-11 flex-shrink-0 items-center justify-between border-b border-border px-3">
              <span className="text-[0.82rem] font-semibold text-ink">Past conversations</span>
              <button onClick={() => setShowHistory(false)} className="icon-btn" aria-label="Close">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto p-2">
              {threads === null ? (
                <div className="skeleton h-10 w-full" />
              ) : threads.length === 0 ? (
                <p className="px-2 py-6 text-center text-[0.8rem] text-ink-dim">
                  Nothing saved yet. Ask something and it will appear here.
                </p>
              ) : (
                threads.map((t) => (
                  <div
                    key={t.id}
                    className={`group flex items-start gap-1.5 rounded-lg px-2 py-2 transition-colors ${
                      t.id === threadId ? "bg-accent/[0.08]" : "hover:bg-surface-hover"
                    }`}
                  >
                    <button onClick={() => open(t.id)} className="min-w-0 flex-1 text-left">
                      <span className="line-clamp-2 text-[0.8rem] leading-snug text-ink">
                        {t.title}
                      </span>
                      <span className="mt-0.5 block text-[0.68rem] text-ink-dim">
                        {t.turn_count} message{t.turn_count === 1 ? "" : "s"}
                        {t.updated_at ? ` · ${new Date(t.updated_at).toLocaleDateString()}` : ""}
                      </span>
                    </button>
                    <button
                      onClick={() => remove(t.id)}
                      className="icon-btn !h-6 !w-6 flex-shrink-0 opacity-0 hover:text-danger group-hover:opacity-100"
                      aria-label="Delete this conversation"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      )}

      {/* ---- messages ---- */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-[760px] flex-col gap-5 p-6">
          {history.length === 0 && !busy && !pending && (
            <div className="pt-6 text-center">
              <span className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-xl bg-accent/[0.12]">
                <Sparkles className="h-5 w-5 text-accent" />
              </span>
              <p className="text-[1.05rem] font-bold tracking-tight text-ink">
                {intro?.title ?? "Ask anything about this video"}
              </p>
              <p className="mx-auto mt-1.5 max-w-[460px] text-[0.87rem] leading-relaxed text-ink-muted">
                {intro?.body ??
                  "Follow-ups keep their context, and every answer cites the moment it came from. If the video doesn't cover it, you still get an answer — marked as going beyond the video."}
              </p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {starters.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="rounded-full border border-border bg-surface-raised px-3.5 py-1.5 text-[0.8rem] text-ink-muted transition-colors hover:border-border-strong hover:text-ink"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {history.map((h, i) => (
            <div key={i} className="flex flex-col gap-3">
              <p className="bubble-user">{h.q}</p>

              <div className="bubble-ai">
                <div className="mb-2 flex flex-wrap gap-1.5">
                  {h.offTopic ? (
                    <span className="pill pill-warning">not about this video</span>
                  ) : (
                    <span
                      className={
                        h.confidence === "high" ? "pill pill-success" : "pill pill-warning"
                      }
                    >
                      {h.confidence} confidence
                    </span>
                  )}
                  {h.beyondVideo && !h.offTopic && (
                    <span className="pill pill-accent">goes beyond the video</span>
                  )}
                </div>

                {h.error ? (
                  <p className="prose-body text-danger">{h.a}</p>
                ) : (
                  <Markdown>{h.a}</Markdown>
                )}

                {!h.error && (
                  <div className="mt-2.5 flex justify-end">
                    <button
                      onClick={() => keep(i, h)}
                      disabled={saved.has(i)}
                      className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[0.75rem] font-medium text-ink-dim transition-colors hover:bg-surface-hover hover:text-ink disabled:text-success disabled:hover:bg-transparent"
                      title="Keep this answer in your notes"
                    >
                      {saved.has(i) ? (
                        <>
                          <Check className="h-3.5 w-3.5" />
                          Saved to notes
                        </>
                      ) : (
                        <>
                          <BookmarkPlus className="h-3.5 w-3.5" />
                          Save to notes
                        </>
                      )}
                    </button>
                  </div>
                )}

                {h.sources?.length > 0 && (
                  <details className="mt-2.5 border-t border-border pt-2">
                    <summary className="cursor-pointer text-[0.78rem] font-medium text-ink-dim">
                      {h.sources.length} cited moment{h.sources.length > 1 ? "s" : ""}
                    </summary>
                    <div className="mt-2 flex flex-col gap-2">
                      {h.sources.map((s: any, k: number) => (
                        <div key={k} className="flex gap-2 text-[0.82rem] text-ink-muted">
                          <Stamp sec={s.start} sourceRef={sourceRef} onSeek={onSeek} />
                          <span className="min-w-0 flex-1">
                            {s.video_title && (
                              <a
                                href={`/v/${s.video_id}`}
                                className="mr-1.5 font-medium text-accent no-underline hover:underline"
                              >
                                {s.video_title}
                              </a>
                            )}
                            <span className="line-clamp-3">
                              {s.text || <em className="text-ink-dim">from the explanation</em>}
                            </span>
                          </span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}

          {pending && (
            <div className="flex flex-col gap-3">
              <p className="bubble-user">{pending}</p>
            </div>
          )}

          {busy && (
            <div className="bubble-ai flex items-center gap-2">
              <span className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent"
                    style={{ animationDelay: `${i * 140}ms` }}
                  />
                ))}
              </span>
              <span className="text-[0.84rem] text-ink-dim">Reading the video&hellip;</span>
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      {/* ---- composer ---- */}
      <div className="flex-shrink-0 border-t border-border p-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(q);
          }}
          className="mx-auto max-w-[760px]"
        >
          <div className="composer">
            <textarea
              ref={boxRef}
              rows={1}
              value={q}
              placeholder="Ask a question, or ask it to go deeper&hellip;"
              onChange={(e) => {
                setQ(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(q);
                }
              }}
            />
            <button
              type="submit"
              disabled={busy || !q.trim()}
              aria-label="Send"
              className="btn btn-primary !h-9 !w-9 !rounded-xl !p-0"
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
