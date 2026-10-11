"use client";

import { Check, RefreshCw, Sparkles, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type QuizQuestion } from "@/lib/api";
import { Stamp } from "@/components/Bits";

const TONE: Record<string, string> = {
  easy: "pill-success",
  medium: "pill-accent",
  hard: "pill-warning",
};

export function Quiz({
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
  const [questions, setQuestions] = useState<QuizQuestion[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [available, setAvailable] = useState(true);
  /** index -> chosen option. Answering is final, as in a real test. */
  const [picked, setPicked] = useState<Record<number, number>>({});

  useEffect(() => {
    api
      .quiz(videoId)
      .then((q) => {
        setQuestions(q.questions?.length ? q.questions : []);
        if (q.available === false) setAvailable(false);
      })
      .catch((e) => setErr(e.message));
  }, [videoId]);

  async function generate(force: boolean) {
    setBusy(true);
    setErr(null);
    try {
      const q = await api.makeQuiz(videoId, 10, force);
      setQuestions(q.questions);
      setPicked({});
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (err && !questions?.length) return <Panel>{<p className="text-danger">{err}</p>}</Panel>;

  if (questions === null) {
    return (
      <Panel>
        <div className="skeleton h-4 w-40" />
      </Panel>
    );
  }

  if (!questions.length) {
    return (
      <Panel>
        <span className="mb-3 grid h-11 w-11 place-items-center rounded-xl bg-accent/[0.12]">
          <Sparkles className="h-5 w-5 text-accent" />
        </span>
        <p className="text-[1.05rem] font-bold tracking-tight text-ink">
          Test what you actually understood
        </p>
        <p className="mt-1.5 max-w-[440px] text-[0.87rem] leading-relaxed text-ink-muted">
          Ten multiple-choice questions written from this video&apos;s own explanation, with
          distractors built from the misconceptions it warns about. Each one links back to the
          moment that covers it.
        </p>
        {!hasUnderstanding ? (
          <p className="mt-4 text-[0.85rem] text-warning">
            This video was processed without a language model, so there is no explanation to
            build questions from.
          </p>
        ) : !available ? (
          <p className="mt-4 text-[0.85rem] text-warning">
            No language model is configured on the server.
          </p>
        ) : (
          <button onClick={() => generate(false)} disabled={busy} className="btn btn-primary mt-5">
            {busy ? "Writing questions…" : "Generate quiz"}
          </button>
        )}
        {err && <p className="mt-3 text-[0.85rem] text-danger">{err}</p>}
      </Panel>
    );
  }

  const answered = Object.keys(picked).length;
  const correct = questions.reduce(
    (n, q, i) => n + (picked[i] === q.correct_index ? 1 : 0),
    0,
  );
  const done = answered === questions.length;

  return (
    <div className="panel-body">
      <div className="mx-auto flex max-w-[720px] flex-col gap-4 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[0.82rem] font-semibold text-ink">
              {answered} of {questions.length} answered
              {answered > 0 && (
                <span className={done ? "ml-2 text-accent" : "ml-2 text-ink-dim"}>
                  · {correct} correct
                </span>
              )}
            </p>
            {done && (
              <p className="mt-0.5 text-[0.78rem] text-ink-dim">
                {Math.round((correct / questions.length) * 100)}% —{" "}
                {correct === questions.length
                  ? "every one."
                  : correct >= questions.length * 0.7
                    ? "solid. Check the ones you missed."
                    : "worth another pass through the explanation."}
              </p>
            )}
          </div>
          <button
            onClick={() => generate(true)}
            disabled={busy}
            className="btn gap-1.5 !px-3 !py-1.5 !text-[0.78rem]"
            title="Write a different set of questions"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} />
            {busy ? "Writing…" : "New set"}
          </button>
        </div>

        <div className="h-1.5 overflow-hidden rounded-full bg-surface-active">
          <div
            className="h-full rounded-full bg-accent transition-all"
            style={{ width: `${(answered / questions.length) * 100}%` }}
          />
        </div>

        {err && <p className="text-[0.85rem] text-danger">{err}</p>}

        {questions.map((q, i) => {
          const choice = picked[i];
          const settled = choice !== undefined;
          return (
            <div key={i} className="rounded-xl border border-border bg-surface-raised p-4">
              <div className="mb-2 flex items-start gap-2.5">
                <span className="mt-0.5 grid h-5 w-5 flex-shrink-0 place-items-center rounded-full bg-accent/[0.12] font-mono text-[0.68rem] font-bold text-accent">
                  {i + 1}
                </span>
                <p className="min-w-0 flex-1 font-semibold leading-snug text-ink">{q.question}</p>
                <span className={`pill flex-shrink-0 ${TONE[q.difficulty] ?? ""}`}>
                  {q.difficulty}
                </span>
              </div>

              <div className="ml-7 flex flex-col gap-1.5">
                {q.options.map((opt, k) => {
                  const isCorrect = k === q.correct_index;
                  const isPicked = choice === k;
                  // Before answering every option is neutral; after, the right
                  // one is always shown, so a wrong pick still teaches.
                  const tone = !settled
                    ? "border-border hover:border-border-strong hover:bg-surface-hover"
                    : isCorrect
                      ? "border-success/40 bg-success/[0.08]"
                      : isPicked
                        ? "border-danger/40 bg-danger/[0.07]"
                        : "border-border opacity-55";
                  return (
                    <button
                      key={k}
                      disabled={settled}
                      onClick={() => setPicked((p) => ({ ...p, [i]: k }))}
                      className={`flex items-start gap-2.5 rounded-lg border px-3 py-2 text-left text-[0.86rem] leading-snug transition-colors ${tone}`}
                    >
                      <span className="mt-[1px] font-mono text-[0.72rem] font-bold text-ink-dim">
                        {"ABCD"[k]}
                      </span>
                      <span className="min-w-0 flex-1 text-ink">{opt}</span>
                      {settled && isCorrect && (
                        <Check className="mt-[1px] h-4 w-4 flex-shrink-0 text-success" />
                      )}
                      {settled && isPicked && !isCorrect && (
                        <X className="mt-[1px] h-4 w-4 flex-shrink-0 text-danger" />
                      )}
                    </button>
                  );
                })}
              </div>

              {settled && (
                <div className="ml-7 mt-2.5 border-t border-border pt-2.5">
                  <div className="flex items-start gap-2">
                    <Stamp sec={q.start_sec} sourceRef={sourceRef} onSeek={onSeek} />
                    <p className="min-w-0 flex-1 text-[0.84rem] leading-relaxed text-ink-muted">
                      {q.explanation}
                    </p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Panel({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid h-full place-items-center p-10">
      <div className="flex max-w-[480px] flex-col items-center text-center">{children}</div>
    </div>
  );
}
