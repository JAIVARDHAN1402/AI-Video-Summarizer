"use client";

import {
  Check,
  ChevronDown,
  Code2,
  Copy,
  Languages,
  RefreshCw,
  Search,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, type Problem, type ProblemDoc } from "@/lib/api";
import { Stamp } from "@/components/Bits";

const TONE: Record<string, string> = {
  easy: "pill-success",
  medium: "pill-accent",
  hard: "pill-warning",
};

/**
 * Every problem a coding video works through, in any language of its family.
 *
 * The list itself is extracted once and cached. The other languages are not:
 * seventy problems in five dialects is three hundred and fifty answers nobody
 * asked for, so a problem is translated the first time it is opened in a
 * language it does not have yet, and all its missing languages are filled in
 * together because that costs one request instead of four.
 */
export function Problems({
  videoId,
  hasUnderstanding,
  sourceRef,
  onSeek,
}: {
  videoId: string;
  hasUnderstanding: boolean;
  sourceRef: string;
  onSeek: (s: number) => void;
}) {
  const [doc, setDoc] = useState<ProblemDoc | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [available, setAvailable] = useState(true);

  const [query, setQuery] = useState("");
  const [difficulty, setDifficulty] = useState<string>("all");
  const [topic, setTopic] = useState<string>("all");
  const [open, setOpen] = useState<number | null>(null);
  /** Which language each problem is being read in. */
  const [lang, setLang] = useState<Record<number, string>>({});
  /** Problems currently being translated, so the tab can say so. */
  const [filling, setFilling] = useState<Record<number, boolean>>({});
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    api
      .problems(videoId)
      .then((d) => {
        setDoc(d);
        if (d.available === false) setAvailable(false);
      })
      .catch((e) => setErr(e.message));
  }, [videoId]);

  const problems = doc?.problems ?? null;
  const targets = doc?.targets ?? [];

  const topics = useMemo(() => {
    const seen = new Set<string>();
    for (const p of problems || []) for (const t of p.topics || []) seen.add(t);
    return [...seen].sort();
  }, [problems]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (problems || []).filter((p) => {
      if (difficulty !== "all" && p.difficulty !== difficulty) return false;
      if (topic !== "all" && !(p.topics || []).includes(topic)) return false;
      if (!q) return true;
      return (
        p.title.toLowerCase().includes(q) ||
        p.statement.toLowerCase().includes(q) ||
        Object.values(p.solutions || {}).some((c) => c.toLowerCase().includes(q))
      );
    });
  }, [problems, query, difficulty, topic]);

  async function extract(force: boolean) {
    setBusy(true);
    setErr(null);
    try {
      setDoc(await api.makeProblems(videoId, force));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  /** Show a problem in `language`, translating it first if it has no such solution. */
  async function pick(p: Problem, language: string) {
    setLang((m) => ({ ...m, [p.n]: language }));
    if (p.solutions?.[language]) return;
    setFilling((m) => ({ ...m, [p.n]: true }));
    try {
      const updated = await api.translateProblem(videoId, p.n);
      setDoc((d) =>
        d ? { ...d, problems: d.problems.map((x) => (x.n === p.n ? updated : x)) } : d,
      );
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setFilling((m) => ({ ...m, [p.n]: false }));
    }
  }

  function copy(key: string, text: string) {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(key);
      setTimeout(() => setCopied((c) => (c === key ? null : c)), 1400);
    });
  }

  if (err && !problems?.length)
    return (
      <Panel>
        <p className="text-danger">{err}</p>
        <button onClick={() => extract(false)} disabled={busy} className="btn mt-4">
          Try again
        </button>
      </Panel>
    );

  if (problems === null)
    return (
      <Panel>
        <div className="skeleton h-4 w-44" />
      </Panel>
    );

  if (!problems.length)
    return (
      <Panel>
        <span className="mb-3 grid h-11 w-11 place-items-center rounded-xl bg-success/[0.12]">
          <Code2 className="h-5 w-5 text-success" />
        </span>
        <p className="text-[1.05rem] font-bold tracking-tight text-ink">
          Every problem this video solves
        </p>
        <p className="mt-1.5 max-w-[460px] text-[0.87rem] leading-relaxed text-ink-muted">
          A practice video works through dozens of questions and then they are gone. This reads
          the whole video — speech and the code that was on screen — and writes each one down
          with its solution, so you can revise the list instead of scrubbing the timeline. Every
          answer can then be shown in any dialect of the same family.
        </p>
        {!hasUnderstanding ? (
          <p className="mt-4 text-[0.85rem] text-warning">
            This video was processed without a language model, so there is no explanation to read.
          </p>
        ) : !available ? (
          <p className="mt-4 text-[0.85rem] text-warning">
            No language model is configured on the server.
          </p>
        ) : (
          <>
            <button
              onClick={() => extract(false)}
              disabled={busy}
              className="btn btn-primary mt-5"
            >
              {busy ? "Reading the video…" : "Extract problems"}
            </button>
            <p className="mt-2.5 text-[0.76rem] text-ink-dim">
              Reads the video in parts, so a long one takes a few minutes.
            </p>
          </>
        )}
        {err && <p className="mt-3 text-[0.85rem] text-danger">{err}</p>}
      </Panel>
    );

  return (
    <div className="panel-body">
      <div className="mx-auto flex max-w-[860px] flex-col gap-4 p-6">
        {/* ---- header: what was found, and how to narrow it ---- */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[1.05rem] font-bold tracking-tight text-ink">
              {problems.length} problems solved
            </p>
            <p className="mt-0.5 text-[0.8rem] text-ink-dim">
              {doc?.primary_language ? `Shown in ${doc.primary_language} as the video wrote them` : ""}
              {targets.length > 1 ? ` · ${targets.length} languages available` : ""}
              {doc?.parts_failed ? ` · ${doc.parts_failed} part(s) could not be read` : ""}
            </p>
          </div>
          <button onClick={() => extract(true)} disabled={busy} className="btn">
            <RefreshCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} />
            {busy ? "Reading…" : "Re-read"}
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="relative flex-1 min-w-[180px]">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-dim" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search questions and code…"
              className="input w-full !pl-8"
            />
          </label>
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="input"
          >
            <option value="all">Any difficulty</option>
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
          {topics.length > 0 && (
            <select value={topic} onChange={(e) => setTopic(e.target.value)} className="input">
              <option value="all">Any topic</option>
              {topics.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          )}
        </div>

        {shown.length === 0 && (
          <p className="py-10 text-center text-[0.87rem] text-ink-dim">
            No problem matches that filter.
          </p>
        )}

        {/* ---- the list ---- */}
        {shown.map((p) => {
          const isOpen = open === p.n;
          const current = lang[p.n] || p.language || targets[0] || "";
          const code = p.solutions?.[current];
          const note = p.notes?.[current];
          return (
            <div key={p.n} className="card overflow-hidden !p-0">
              <div className="flex items-start gap-3 p-4 transition-colors hover:bg-surface-hover">
                <button
                  onClick={() => setOpen(isOpen ? null : p.n)}
                  aria-expanded={isOpen}
                  className="flex min-w-0 flex-1 items-start gap-3 text-left"
                >
                  <span className="mt-0.5 grid h-6 w-6 flex-shrink-0 place-items-center rounded-md bg-surface-hover font-mono text-[0.72rem] tabular-nums text-ink-muted">
                    {p.n}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[0.92rem] font-semibold text-ink">{p.title}</span>
                    <span className="mt-1 flex flex-wrap items-center gap-1.5">
                      <span className={`pill ${TONE[p.difficulty] || "pill-accent"}`}>
                        {p.difficulty}
                      </span>
                      {(p.topics || []).map((t) => (
                        <span key={t} className="pill">
                          {t}
                        </span>
                      ))}
                    </span>
                  </span>
                </button>
                <div className="flex flex-shrink-0 items-center gap-2">
                  <Stamp sec={p.start_sec} sourceRef={sourceRef} onSeek={onSeek} />
                  <button
                    onClick={() => setOpen(isOpen ? null : p.n)}
                    aria-label={isOpen ? "Collapse" : "Expand"}
                    className="text-ink-dim transition-colors hover:text-ink"
                  >
                    <ChevronDown
                      className={`h-4 w-4 transition-transform ${isOpen ? "rotate-180" : ""}`}
                    />
                  </button>
                </div>
              </div>

              {isOpen && (
                <div className="border-t border-line px-4 pb-4 pt-3">
                  <p className="text-[0.88rem] leading-relaxed text-ink">{p.statement}</p>
                  {p.approach && (
                    <p className="mt-2 text-[0.84rem] leading-relaxed text-ink-muted">
                      <span className="font-semibold text-ink">Approach — </span>
                      {p.approach}
                    </p>
                  )}

                  {targets.length > 0 && (
                    <div className="mt-3 flex flex-wrap items-center gap-1.5">
                      <Languages className="h-3.5 w-3.5 text-ink-dim" />
                      {targets.map((t) => (
                        <button
                          key={t}
                          onClick={() => pick(p, t)}
                          className={`rounded-md px-2 py-1 text-[0.76rem] font-semibold transition-colors ${
                            current === t
                              ? "bg-accent/[0.12] text-accent"
                              : "text-ink-dim hover:bg-surface-hover hover:text-ink"
                          }`}
                          title={
                            p.solutions?.[t]
                              ? `Show in ${t}`
                              : `Translate into ${t} (fills every missing language at once)`
                          }
                        >
                          {t}
                          {!p.solutions?.[t] && (
                            <span className="ml-1 text-ink-dim" aria-hidden>
                              +
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  )}

                  <div className="relative mt-2">
                    {filling[p.n] && !code ? (
                      <div className="card flex items-center gap-2 !bg-surface-hover">
                        <Sparkles className="h-3.5 w-3.5 animate-pulse text-accent" />
                        <span className="text-[0.84rem] text-ink-muted">
                          Rewriting it in {current}…
                        </span>
                      </div>
                    ) : code ? (
                      <>
                        <pre className="overflow-x-auto rounded-lg border border-line bg-surface-hover p-3 font-mono text-[0.78rem] leading-relaxed text-ink">
                          {code}
                        </pre>
                        <button
                          onClick={() => copy(`${p.n}-${current}`, code)}
                          className="absolute right-2 top-2 rounded-md bg-surface p-1.5 text-ink-dim transition-colors hover:text-ink"
                          title="Copy"
                        >
                          {copied === `${p.n}-${current}` ? (
                            <Check className="h-3.5 w-3.5 text-success" />
                          ) : (
                            <Copy className="h-3.5 w-3.5" />
                          )}
                        </button>
                      </>
                    ) : (
                      <p className="text-[0.84rem] text-ink-dim">
                        No solution recorded in {current} yet — pick the language above to write
                        one.
                      </p>
                    )}
                  </div>

                  {note && (
                    <p className="mt-2 text-[0.8rem] leading-relaxed text-warning">
                      {current} differs here: {note}
                    </p>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {err && <p className="text-[0.85rem] text-danger">{err}</p>}
      </div>
    </div>
  );
}

function Panel({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid h-full place-items-center p-10">
      <div className="flex max-w-[500px] flex-col items-center text-center">{children}</div>
    </div>
  );
}
