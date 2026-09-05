"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { api, type Profile, type VideoType } from "@/lib/api";

const VIDEO_TYPES: { value: VideoType; label: string }[] = [
  { value: "lecture", label: "Lecture / tutorial" },
  { value: "coding", label: "Coding / algorithms" },
  { value: "meeting", label: "Meeting recording" },
  { value: "interview", label: "Interview / podcast" },
  { value: "talk", label: "Conference talk" },
  { value: "news", label: "News segment" },
  { value: "documentary", label: "Documentary / explainer" },
  { value: "other", label: "Other" },
];

const PROFILES: { value: Profile; label: string; hint: string }[] = [
  { value: "fast", label: "Fast", hint: "Quick pass, lighter analysis" },
  { value: "balanced", label: "Balanced", hint: "Full explanation, ~5 min target" },
  { value: "accurate", label: "Accurate", hint: "Largest models, no time limit" },
];

export default function Home() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [videoType, setVideoType] = useState<VideoType>("lecture");
  const [topic, setTopic] = useState("");
  const [profile, setProfile] = useState<Profile>("balanced");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const { id } = await api.createJob({
        url: url.trim(),
        video_type: videoType,
        topic: topic.trim(),
        profile,
      });
      router.push(`/jobs/${id}`);
    } catch (err: any) {
      setError(err.message || "Could not start the job.");
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="mx-auto max-w-[620px] pb-8 pt-6 text-center">
        <h1 className="bg-gradient-to-br from-white via-[#C3C2FD] to-[#9B8AF5] bg-clip-text text-[2.6rem] font-extrabold leading-[1.1] tracking-tight text-transparent">
          Understand any video
          <br />
          without watching it
        </h1>
        <p className="mx-auto mt-3 max-w-[520px] text-[1rem] leading-relaxed text-ink-muted">
          Paste a link. You get a teachable explanation, section by section, with the
          algorithm and code for programming videos — and an assistant that cites the
          moment it got each answer.
        </p>
      </div>

      <form onSubmit={submit} className="card mx-auto max-w-[760px] p-6">
        <label className="label" htmlFor="url">Video URL</label>
        <input
          id="url"
          className="input"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          autoFocus
        />

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-[1fr_1.4fr]">
          <div>
            <label className="label" htmlFor="type">What kind of video is this?</label>
            <select
              id="type"
              className="select"
              value={videoType}
              onChange={(e) => setVideoType(e.target.value as VideoType)}
            >
              {VIDEO_TYPES.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="topic">Topic (optional)</label>
            <input
              id="topic"
              className="input"
              placeholder="e.g. Backtracking, Transformers, Q3 roadmap"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
            />
          </div>
        </div>

        <div className="mt-4">
          <span className="label">Processing profile</span>
          <div className="grid grid-cols-3 gap-2">
            {PROFILES.map((p) => (
              <button
                type="button"
                key={p.value}
                onClick={() => setProfile(p.value)}
                className={`rounded-sm border px-3 py-2 text-left text-[0.8rem] transition-colors ${
                  profile === p.value
                    ? "border-accent bg-accent/[0.12] text-ink"
                    : "border-border bg-surface text-ink-muted hover:border-border-strong"
                }`}
              >
                <div className="font-semibold text-ink">{p.label}</div>
                <div className="mt-0.5 text-[0.7rem] text-ink-dim">{p.hint}</div>
              </button>
            ))}
          </div>
        </div>

        {error && (
          <p className="mt-4 rounded-sm border border-danger/40 bg-danger/[0.1] px-3 py-2 text-[0.85rem] text-danger">
            {error}
          </p>
        )}

        <button className="btn btn-primary mt-5 w-full" disabled={busy || !url.trim()}>
          {busy ? "Starting…" : "Summarize video"}
        </button>
      </form>

      <div className="mx-auto mt-4 flex max-w-[760px] flex-wrap justify-center gap-2">
        {["Teachable explanation", "Auto chapters", "Algorithm + multi-language code", "Scene extraction", "Cited Q&A"].map(
          (f) => (
            <span key={f} className="pill">{f}</span>
          ),
        )}
      </div>
    </div>
  );
}
