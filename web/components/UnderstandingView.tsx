"use client";

import type { Understanding } from "@/lib/api";
import { CodeTabs, SectionHead, Stamp } from "./Bits";

/**
 * Renders the LLM understanding top to bottom: overview -> the section-by-
 * section teaching -> recap -> reference material (key points, glossary) ->
 * per-type extras (algorithm+code, meeting decisions) -> on-screen material.
 */
export function UnderstandingView({
  u,
  sourceRef,
  onSeek,
}: {
  u: Understanding;
  sourceRef?: string;
  onSeek?: (s: number) => void;
}) {
  return (
    <div className="flex flex-col gap-7">
      {u.overview && (
        <section>
          <SectionHead title="Overview" />
          <p className="prose-body">{u.overview}</p>
        </section>
      )}

      {u.explanation?.length > 0 && (
        <section>
          <SectionHead
            title="Explanation"
            hint={`${u.explanation.length} sections · taught step by step, from the video and beyond it`}
          />
          <div className="flex flex-col">
            {u.explanation.map((b, i) => (
              <div key={i} className="border-b border-border py-4 last:border-0">
                <div className="mb-1.5 flex flex-wrap items-center gap-2.5">
                  <span className="flex h-5 w-5 items-center justify-center rounded-xs border border-border bg-surface-raised font-mono text-[0.7rem] font-bold text-ink-dim">
                    {i + 1}
                  </span>
                  <Stamp sec={b.start_sec} sourceRef={sourceRef} onSeek={onSeek} />
                  <span className="text-[0.98rem] font-semibold tracking-tight text-ink">
                    {b.heading}
                  </span>
                  {!b.from_video && (
                    <span className="rounded-full border border-cyan/30 bg-cyan/[0.12] px-2 py-0.5 text-[0.62rem] font-bold uppercase tracking-wide text-cyan">
                      added context
                    </span>
                  )}
                </div>
                <p className="prose-body whitespace-pre-wrap">{b.body}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {u.summary && (
        <section>
          <SectionHead title="In summary" hint="The whole thing, briefly" />
          <div className="rounded border border-accent/25 bg-gradient-to-b from-accent/[0.14] to-transparent p-4 text-[0.93rem] leading-relaxed text-ink">
            {u.summary}
          </div>
        </section>
      )}

      {u.key_points?.length > 0 && (
        <section>
          <SectionHead title="Key points" />
          <ul className="ml-5 list-disc space-y-1.5 text-[0.9rem] leading-relaxed text-ink">
            {u.key_points.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </section>
      )}

      {u.coding && (u.coding.algorithm_name || u.coding.implementations?.length) && (
        <section>
          <SectionHead
            title="Algorithm & code"
            hint="Derived from the video, completed and corrected"
          />
          {u.coding.algorithm_name && (
            <p className="mb-1 font-semibold text-ink">{u.coding.algorithm_name}</p>
          )}
          {u.coding.intuition && <p className="prose-body mb-2">{u.coding.intuition}</p>}
          {u.coding.steps?.length > 0 && (
            <ol className="mb-2 ml-5 list-decimal space-y-1 text-[0.88rem] leading-relaxed text-ink">
              {u.coding.steps.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ol>
          )}
          {(u.coding.time_complexity || u.coding.space_complexity) && (
            <p className="mb-3 text-[0.85rem] text-ink-muted">
              {u.coding.time_complexity && (
                <>
                  <strong className="text-ink">Time:</strong> {u.coding.time_complexity}
                </>
              )}
              {u.coding.time_complexity && u.coding.space_complexity && " · "}
              {u.coding.space_complexity && (
                <>
                  <strong className="text-ink">Space:</strong> {u.coding.space_complexity}
                </>
              )}
            </p>
          )}
          <CodeTabs impls={u.coding.implementations || []} />
        </section>
      )}

      {u.meeting && (u.meeting.decisions?.length || u.meeting.action_items?.length) && (
        <section className="flex flex-col gap-4">
          {u.meeting.decisions?.length > 0 && (
            <div>
              <SectionHead title="Decisions" />
              <ul className="ml-5 list-disc space-y-1.5 text-[0.9rem] text-ink">
                {u.meeting.decisions.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>
          )}
          {u.meeting.action_items?.length > 0 && (
            <div>
              <SectionHead title="Action items" />
              <ul className="ml-5 list-disc space-y-1.5 text-[0.9rem] text-ink">
                {u.meeting.action_items.map((a, i) => (
                  <li key={i}>
                    {a.owner && <strong className="text-ink">{a.owner} — </strong>}
                    {a.task}
                    {a.due && <span className="text-ink-dim"> (due {a.due})</span>}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {u.glossary?.length > 0 && (
        <section>
          <SectionHead title="Glossary" />
          <div className="flex flex-col">
            {u.glossary.map((g, i) => (
              <div
                key={i}
                className="flex gap-3 border-b border-border py-2 text-[0.885rem] leading-relaxed last:border-0"
              >
                <span className="min-w-[150px] flex-shrink-0 font-semibold text-ink">
                  {g.term}
                </span>
                <span className="text-ink-muted">{g.definition}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {u.scene_notes?.length > 0 && (
        <section>
          <SectionHead
            title="On-screen material"
            hint="Slides, diagrams and boards, explained"
          />
          <div className="flex flex-col gap-4">
            {u.scene_notes.map((n, i) => (
              <div key={i} className="grid grid-cols-1 gap-3 sm:grid-cols-[240px_1fr]">
                {n.keyframe_url && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={n.keyframe_url}
                    alt=""
                    loading="lazy"
                    className="w-full rounded-sm border border-border object-cover"
                  />
                )}
                <div>
                  <Stamp sec={n.start} sourceRef={sourceRef} onSeek={onSeek} />
                  <p className="prose-body mt-1">{n.what_it_shows}</p>
                  {n.on_screen_text && (
                    <details className="mt-1">
                      <summary className="cursor-pointer text-[0.78rem] text-ink-dim">
                        Text on screen
                      </summary>
                      <p className="mt-1 whitespace-pre-wrap text-[0.78rem] text-ink-dim">
                        {n.on_screen_text}
                      </p>
                    </details>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
