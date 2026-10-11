"use client";

import { ArrowUp, BookmarkPlus, Check, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { Stamp } from "@/components/Bits";

export type Turn = {
  q: string;
  a: string;
  confidence: string;
  sources: { start: number; end: number; text: string }[];
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
  sourceRef,
  onSeek,
}: {
  videoId: string;
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<Turn[]>([]);
  const [saved, setSaved] = useState<Set<number>>(new Set());
  const endRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [history, busy]);

  async function keep(i: number, t: Turn) {
    try {
      await api.addNote(
        videoId,
        `**${t.q}**

${t.a}`,
        t.sources?.[0]?.start,
        "chat",
      );
      setSaved((s) => new Set(s).add(i));
    } catch {
      /* the Notes tab is the source of truth; a failed save is visible there */
    }
  }

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    setQ("");
    setBusy(true);
    try {
      // The thread so far goes with the question, so "explain that further"
      // resolves against it rather than being read as a standalone query.
      const prior = history.filter((t) => !t.error).map(({ q, a }) => ({ q, a }));
      const r = await api.ask(videoId, question, prior);
      setHistory((h) => [
        ...h,
        {
          q: question,
          a: r.answer,
          confidence: r.confidence,
          sources: r.sources,
          beyondVideo: r.beyond_video,
          offTopic: r.off_topic,
        },
      ]);
    } catch (err: any) {
      setHistory((h) => [
        ...h,
        { q: question, a: `Error: ${err.message}`, confidence: "low", sources: [], error: true },
      ]);
    } finally {
      setBusy(false);
      boxRef.current?.focus();
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-[760px] flex-col gap-5 p-6">
          {history.length === 0 && !busy && (
            <div className="pt-8 text-center">
              <span className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-xl bg-accent/[0.12]">
                <Sparkles className="h-5 w-5 text-accent" />
              </span>
              <p className="text-[1.05rem] font-bold tracking-tight text-ink">
                Ask anything about this video
              </p>
              <p className="mx-auto mt-1.5 max-w-[460px] text-[0.87rem] leading-relaxed text-ink-muted">
                Follow-ups keep their context, and every answer cites the moment it came from.
                If the video doesn&apos;t cover it, you still get an answer &mdash; marked as
                going beyond the video.
              </p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {STARTERS.map((s) => (
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
                          <span className="line-clamp-3">{s.text}</span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}

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
