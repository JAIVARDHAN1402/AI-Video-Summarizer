"use client";

import { Layers, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api, type Notebook } from "@/lib/api";

export default function Notebooks() {
  const router = useRouter();
  const [books, setBooks] = useState<Notebook[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.notebooks().then(setBooks).catch((e) => setErr(e.message));
  }, []);

  async function create() {
    setBusy(true);
    try {
      const nb = await api.createNotebook("Untitled notebook");
      router.push(`/n/${nb.id}`);
    } catch (e: any) {
      setErr(e.message);
      setBusy(false);
    }
  }

  async function remove(id: string) {
    const before = books ?? [];
    setBooks(before.filter((b) => b.id !== id));
    try {
      await api.deleteNotebook(id);
    } catch (e: any) {
      setErr(e.message);
      setBooks(before);
    }
  }

  return (
    <div className="mx-auto h-full max-w-[900px] overflow-y-auto px-5 py-8">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="m-0 text-[1.45rem] font-bold tracking-tight text-ink">Notebooks</h1>
          <p className="mt-1 max-w-[520px] text-[0.85rem] leading-relaxed text-ink-dim">
            A notebook is several processed videos treated as one body of material. Questions
            are answered across all of them, and each answer says which video it came from.
          </p>
        </div>
        <button onClick={create} disabled={busy} className="btn btn-primary gap-1.5">
          <Plus className="h-4 w-4" />
          New notebook
        </button>
      </div>

      {err && <p className="mb-4 text-[0.9rem] text-danger">{err}</p>}

      {books === null ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {[0, 1].map((i) => (
            <div key={i} className="card p-4">
              <div className="skeleton h-4 w-2/3" />
              <div className="skeleton mt-2 h-3 w-1/3" />
            </div>
          ))}
        </div>
      ) : books.length === 0 ? (
        <div className="card grid place-items-center p-12 text-center">
          <Layers className="mb-3 h-7 w-7 text-ink-dim" />
          <p className="font-semibold text-ink">No notebooks yet</p>
          <p className="mt-1 max-w-[380px] text-[0.85rem] text-ink-dim">
            Make one to ask a question across a playlist, a module, or a term&apos;s worth of
            lectures at once.
          </p>
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {books.map((b) => {
            const live = b.videos.filter((v) => !v.missing);
            return (
              <div
                key={b.id}
                className="card group flex flex-col gap-2 p-4 transition-all hover:border-border-strong hover:shadow-card-hover"
              >
                <div className="flex items-start gap-2">
                  <Link
                    href={`/n/${b.id}`}
                    className="min-w-0 flex-1 font-semibold text-ink no-underline"
                  >
                    {b.title}
                  </Link>
                  <button
                    onClick={() => remove(b.id)}
                    className="icon-btn !h-6 !w-6 flex-shrink-0 opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
                    title="Delete this notebook"
                    aria-label="Delete this notebook"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>

                <Link href={`/n/${b.id}`} className="no-underline">
                  <p className="text-[0.78rem] text-ink-dim">
                    {live.length} source{live.length === 1 ? "" : "s"}
                  </p>
                  {live.length > 0 && (
                    <ul className="m-0 mt-1.5 list-none p-0">
                      {live.slice(0, 3).map((v) => (
                        <li
                          key={v.video_id}
                          className="truncate text-[0.76rem] leading-relaxed text-ink-muted"
                        >
                          {v.title}
                        </li>
                      ))}
                      {live.length > 3 && (
                        <li className="text-[0.76rem] text-ink-dim">
                          and {live.length - 3} more
                        </li>
                      )}
                    </ul>
                  )}
                </Link>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
