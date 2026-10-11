"use client";

import { AudioLines, Download, Pause, Play, RefreshCw, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, mmss, type AudioOverviewDoc } from "@/lib/api";

export function AudioOverview({
  videoId,
  hasUnderstanding,
}: {
  videoId: string;
  hasUnderstanding: boolean;
}) {
  const [doc, setDoc] = useState<AudioOverviewDoc | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [available, setAvailable] = useState(true);
  const [at, setAt] = useState(0);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const activeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .audioOverview(videoId)
      .then((d) => {
        setDoc(d.turns?.length ? d : { turns: [] });
        if (d.available === false) setAvailable(false);
      })
      .catch((e) => setErr(e.message));
  }, [videoId]);

  // Follow the spoken line, but only while playing — otherwise reading ahead
  // would be yanked back to the playhead on every tick.
  useEffect(() => {
    if (playing) activeRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [playing, at]);

  async function generate(force: boolean) {
    setBusy(true);
    setErr(null);
    try {
      setDoc(await api.makeAudioOverview(videoId, force));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (doc === null) {
    return (
      <div className="p-6">
        <div className="skeleton h-4 w-48" />
      </div>
    );
  }

  if (!doc.turns.length) {
    return (
      <div className="grid h-full place-items-center p-10">
        <div className="flex max-w-[480px] flex-col items-center text-center">
          <span className="mb-3 grid h-11 w-11 place-items-center rounded-xl bg-violet/[0.12]">
            <AudioLines className="h-5 w-5 text-violet" />
          </span>
          <p className="text-[1.05rem] font-bold tracking-tight text-ink">
            Hear two people work through it
          </p>
          <p className="mt-1.5 text-[0.87rem] leading-relaxed text-ink-muted">
            A conversation written from this video&apos;s explanation and spoken in two voices —
            one asking the questions a learner actually has, one answering them. Roughly five
            minutes, and you can read along as it plays.
          </p>
          {!hasUnderstanding ? (
            <p className="mt-4 text-[0.85rem] text-warning">
              This video was processed without a language model, so there is no explanation to
              build a conversation from.
            </p>
          ) : !available ? (
            <p className="mt-4 text-[0.85rem] text-warning">
              This needs a language model and ffmpeg on the server; one of them is missing.
            </p>
          ) : (
            <button
              onClick={() => generate(false)}
              disabled={busy}
              className="btn btn-primary mt-5"
            >
              {busy ? "Writing and speaking… (about a minute)" : "Generate audio overview"}
            </button>
          )}
          {err && <p className="mt-3 text-[0.85rem] text-danger">{err}</p>}
        </div>
      </div>
    );
  }

  const current = doc.turns.findIndex((t) => at >= t.start_sec && at < t.end_sec);
  const audio = audioRef.current;

  return (
    <div className="flex h-full flex-col">
      {/* ---- player ---- */}
      <div className="flex-shrink-0 border-b border-border p-4">
        <div className="mx-auto flex max-w-[720px] items-center gap-3">
          <button
            onClick={() => (playing ? audio?.pause() : audio?.play())}
            className="btn btn-primary !h-11 !w-11 !rounded-full !p-0"
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
          </button>

          <div className="min-w-0 flex-1">
            <input
              type="range"
              min={0}
              max={doc.duration_sec || 0}
              step={0.1}
              value={at}
              onChange={(e) => {
                const v = Number(e.target.value);
                setAt(v);
                if (audio) audio.currentTime = v;
              }}
              className="w-full accent-[rgb(var(--c-accent))]"
              aria-label="Seek"
            />
            <div className="mt-0.5 flex justify-between font-mono text-[0.72rem] tabular-nums text-ink-dim">
              <span>{mmss(at)}</span>
              <span>{mmss(doc.duration_sec || 0)}</span>
            </div>
          </div>

          <button
            onClick={() => {
              if (audio) audio.currentTime = Math.max(0, audio.currentTime - 10);
            }}
            className="icon-btn"
            title="Back 10 seconds"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
          <a
            href={doc.url}
            download="audio-overview.mp3"
            className="icon-btn"
            title="Download the audio"
          >
            <Download className="h-4 w-4" />
          </a>
          <button
            onClick={() => generate(true)}
            disabled={busy}
            className="icon-btn"
            title="Write a different conversation"
          >
            <RefreshCw className={`h-4 w-4 ${busy ? "animate-spin" : ""}`} />
          </button>
        </div>

        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <audio
          ref={audioRef}
          src={doc.url}
          onTimeUpdate={(e) => setAt(e.currentTarget.currentTime)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          className="hidden"
        />
        {err && <p className="mt-2 text-center text-[0.82rem] text-danger">{err}</p>}
      </div>

      {/* ---- transcript, following the voice ---- */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-[720px] flex-col gap-2.5 p-6">
          {doc.turns.map((t, i) => {
            const isA = t.speaker === "HOST_A";
            const live = i === current;
            return (
              <div
                key={i}
                ref={live ? activeRef : undefined}
                onClick={() => {
                  if (audio) audio.currentTime = t.start_sec;
                  setAt(t.start_sec);
                }}
                className={`flex cursor-pointer gap-3 rounded-xl border p-3 transition-colors ${
                  live
                    ? "border-accent/40 bg-accent/[0.06]"
                    : "border-transparent hover:bg-surface-hover"
                }`}
              >
                <span
                  className={`grid h-7 w-7 flex-shrink-0 place-items-center rounded-full text-[0.68rem] font-bold ${
                    isA ? "bg-accent/[0.14] text-accent" : "bg-violet/[0.14] text-violet"
                  }`}
                >
                  {isA ? "A" : "B"}
                </span>
                <div className="min-w-0 flex-1">
                  <p
                    className={`text-[0.89rem] leading-relaxed ${
                      live ? "text-ink" : "text-ink-muted"
                    }`}
                  >
                    {t.text}
                  </p>
                </div>
                <span className="flex-shrink-0 font-mono text-[0.68rem] tabular-nums text-ink-dim">
                  {mmss(t.start_sec)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
