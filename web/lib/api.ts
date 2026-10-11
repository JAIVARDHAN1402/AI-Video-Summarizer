// Typed client for the FastAPI backend. All calls go through /api/* which
// next.config.ts proxies to the backend in dev.

export type VideoType =
  | "lecture" | "coding" | "meeting" | "interview" | "talk" | "news" | "documentary" | "other";
export type Profile = "fast" | "balanced" | "accurate";

export interface JobView {
  id: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  stage: string;
  progress: number;
  message: string;
  error: string | null;
  video_id: string | null;
  title: string | null;
  duration_sec: number | null;
  video_type: string;
  topic: string;
  profile: string;
  routed_worker_class: string | null;
  predicted_sec: number | null;
  sla_target_sec: number | null;
  sla_met: boolean | null;
  timings: Record<string, number>;
  worker_class: string | null;
  elapsed_sec: number | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

/** A saved conversation. The list view omits `turns`. */
export interface ChatThread {
  id: string;
  scope: string;
  title: string;
  turn_count: number;
  turns?: any[];
  created_at: string | null;
  updated_at: string | null;
}

export interface NotebookVideo {
  video_id: string;
  title: string;
  duration_sec: number | null;
  source_ref: string | null;
  /** True when the cache no longer holds this video. */
  missing: boolean;
}
export interface Notebook {
  id: string;
  title: string;
  video_ids: string[];
  videos: NotebookVideo[];
  created_at: string | null;
  updated_at: string | null;
}

/** A reader's own note. The only thing in the cache the pipeline never writes. */
export interface Note {
  id: string;
  text: string;
  created_at: number;
  edited_at?: number;
  start_sec: number | null;
  source: "manual" | "chat" | "explanation";
}

export interface AudioTurn {
  speaker: "HOST_A" | "HOST_B";
  text: string;
  start_sec: number;
  end_sec: number;
}
export interface AudioOverviewDoc {
  turns: AudioTurn[];
  /** Absent until an overview has actually been built. */
  url?: string;
  duration_sec?: number;
  model?: string;
  generated_in_sec?: number;
  /** false when the server lacks a model or ffmpeg. */
  available?: boolean;
}

export interface QuizQuestion {
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
  start_sec: number;
  difficulty: "easy" | "medium" | "hard";
}
export interface QuizDoc {
  questions: QuizQuestion[];
  model?: string;
  generated_in_sec?: number;
  /** false when the server has no language model configured at all. */
  available?: boolean;
}

/** A row in the library: what the cache knows about one processed video. */
export interface VideoSummary {
  video_id: string;
  title: string;
  duration_sec: number | null;
  source_ref: string | null;
  video_type: string | null;
  processed_sec: number | null;
  has_understanding: boolean;
  has_highlight: boolean;
  mtime: number;
}

export interface ExplanationBlock {
  heading: string;
  start_sec: number;
  body: string;
  from_video: boolean;
}
export interface Chapter { start: number; title: string; summary: string }
export interface SceneNote {
  start: number;
  what_it_shows: string;
  keyframe_url?: string;
  on_screen_text?: string;
}
export interface CodeImpl { language: string; code: string }
export interface Understanding {
  video_type: string;
  topic: string;
  model: string;
  overview: string;
  explanation: ExplanationBlock[];
  summary: string;
  key_points: string[];
  glossary: { term: string; definition: string }[];
  chapters: Chapter[];
  scene_notes: SceneNote[];
  coding?: {
    algorithm_name: string;
    intuition: string;
    steps: string[];
    time_complexity: string;
    space_complexity: string;
    implementations: CodeImpl[];
  };
  meeting?: {
    decisions: string[];
    action_items: { owner?: string; task: string; due?: string }[];
  };
}

/** One prior turn, sent back so follow-ups resolve against the conversation. */
export interface AskTurn {
  q: string;
  a: string;
}

export interface AskAnswer {
  answer: string;
  confidence: string;
  sources: { start: number; end: number; text: string }[];
  /** The answer relies on knowledge the video did not cover. */
  beyond_video: boolean;
  /** The question is unrelated to this video's subject. */
  off_topic: boolean;
}

export interface VideoResult {
  video_id: string;
  meta: Record<string, any>;
  understanding: Understanding | null;
  summary: Record<string, any> | null;
  scenes: { scenes: any[]; importance_ranked?: number[] } | null;
  transcript: { segments: { start: number; end: number; text: string }[]; language?: string } | null;
  state: Record<string, any>;
  highlight: { url: string; seconds: number | null; scene_ids: number[] } | null;
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body.slice(0, 300)}` : ""}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createJob: (body: {
    url?: string; upload_id?: string;
    video_type: VideoType; topic: string; profile: Profile; force?: boolean;
  }) =>
    fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<{ id: string; status: string }>(r)),

  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/uploads", { method: "POST", body: form }).then((r) =>
      j<{ upload_id: string; filename: string; bytes: number }>(r),
    );
  },

  job: (id: string) => fetch(`/api/jobs/${id}`).then((r) => j<JobView>(r)),

  cancelJob: (id: string) =>
    fetch(`/api/jobs/${id}/cancel`, { method: "POST" }).then((r) => j<JobView>(r)),

  recentJobs: (limit = 30) =>
    fetch(`/api/jobs?limit=${limit}`).then((r) => j<JobView[]>(r)),

  /** Every processed video on disk — the library's source of truth, since a
   *  video can exist in the cache without a job row behind it. */
  videos: () => fetch("/api/videos").then((r) => j<VideoSummary[]>(r)),

  deleteVideo: async (videoId: string) => {
    const r = await fetch(`/api/videos/${videoId}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`Delete failed (${r.status})`);
  },

  /** Irreversible. The server refuses without the confirm token. */
  clearLibrary: () =>
    fetch("/api/videos?confirm=DELETE", { method: "DELETE" }).then((r) =>
      j<{ removed: number; video_ids: string[] }>(r),
    ),

  video: (videoId: string) =>
    fetch(`/api/videos/${videoId}`).then((r) => j<VideoResult>(r)),

  /** scope is "video:<id>" or "notebook:<id>". */
  chats: (scope: string) =>
    fetch(`/api/chats?scope=${encodeURIComponent(scope)}`).then((r) => j<ChatThread[]>(r)),

  chat: (id: string) => fetch(`/api/chats/${id}`).then((r) => j<ChatThread>(r)),

  createChat: (scope: string, title: string, turns: any[] = []) =>
    fetch("/api/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scope, title, turns }),
    }).then((r) => j<ChatThread>(r)),

  saveChat: (id: string, patch: { title?: string; turns?: any[] }) =>
    fetch(`/api/chats/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).then((r) => j<ChatThread>(r)),

  deleteChat: async (id: string) => {
    const r = await fetch(`/api/chats/${id}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`Delete failed (${r.status})`);
  },

  notebooks: () => fetch("/api/notebooks").then((r) => j<Notebook[]>(r)),

  notebook: (id: string) => fetch(`/api/notebooks/${id}`).then((r) => j<Notebook>(r)),

  createNotebook: (title: string, videoIds: string[] = []) =>
    fetch("/api/notebooks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, video_ids: videoIds }),
    }).then((r) => j<Notebook>(r)),

  /** Omit a field to leave it alone — a rename need not resend membership. */
  updateNotebook: (id: string, patch: { title?: string; video_ids?: string[] }) =>
    fetch(`/api/notebooks/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).then((r) => j<Notebook>(r)),

  deleteNotebook: async (id: string) => {
    const r = await fetch(`/api/notebooks/${id}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`Delete failed (${r.status})`);
  },

  askNotebook: (id: string, question: string, history: AskTurn[] = []) =>
    fetch(`/api/notebooks/${id}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    }).then((r) => j<AskAnswer>(r)),

  notes: (videoId: string) =>
    fetch(`/api/videos/${videoId}/notes`).then((r) => j<Note[]>(r)),

  addNote: (videoId: string, text: string, startSec?: number, source = "manual") =>
    fetch(`/api/videos/${videoId}/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, start_sec: startSec ?? null, source }),
    }).then((r) => j<Note>(r)),

  editNote: (videoId: string, noteId: string, text: string) =>
    fetch(`/api/videos/${videoId}/notes/${noteId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then((r) => j<Note>(r)),

  deleteNote: async (videoId: string, noteId: string) => {
    const r = await fetch(`/api/videos/${videoId}/notes/${noteId}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`Delete failed (${r.status})`);
  },

  audioOverview: (videoId: string) =>
    fetch(`/api/videos/${videoId}/audio-overview`).then((r) => j<AudioOverviewDoc>(r)),

  /** Writes the script and speaks it; takes around a minute. */
  makeAudioOverview: (videoId: string, force = false) =>
    fetch(`/api/videos/${videoId}/audio-overview?force=${force}`, { method: "POST" }).then(
      (r) => j<AudioOverviewDoc>(r),
    ),

  quiz: (videoId: string) =>
    fetch(`/api/videos/${videoId}/quiz`).then((r) => j<QuizDoc>(r)),

  /** `force` writes a fresh set over the cached one. */
  makeQuiz: (videoId: string, count = 10, force = false) =>
    fetch(`/api/videos/${videoId}/quiz?count=${count}&force=${force}`, {
      method: "POST",
    }).then((r) => j<QuizDoc>(r)),

  ask: (videoId: string, question: string, history: AskTurn[] = []) =>
    fetch(`/api/videos/${videoId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    }).then((r) => j<AskAnswer>(r)),

  schedulerClasses: () =>
    fetch("/api/scheduler/classes").then((r) => j<any[]>(r)),
  schedulerSla: () => fetch("/api/scheduler/sla").then((r) => j<any>(r)),
};

export function mmss(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${m}:${String(ss).padStart(2, "0")}`;
}

export function humanDuration(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return s ? `${m} min ${s}s` : `${m} min`;
  }
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec % 3600) / 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

export function ytId(url: string): string | null {
  const m = url.match(
    /(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|shorts\/|embed\/|v\/))([A-Za-z0-9_-]{11})/,
  );
  return m ? m[1] : null;
}
