"use client";

import { PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

export type Side = {
  /** Shown in the panel header and, rotated, on the collapsed rail. */
  title: string;
  icon: ReactNode;
  /** Rendered inside the scrolling body. */
  children: ReactNode;
  /** Optional controls pinned to the right of the header. */
  actions?: ReactNode;
};

/** Remembered per browser so a reader's chosen layout survives a reload.
 *  Never fatal: a private window or blocked storage just means defaults. */
function useRemembered(key: string, fallback: boolean) {
  const [value, setValue] = useState(fallback);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw !== null) setValue(raw === "1");
    } catch {
      /* storage unavailable — keep the default */
    }
    setReady(true);
  }, [key]);

  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(key, value ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [key, value, ready]);

  return [value, setValue] as const;
}

function Rail({
  title,
  icon,
  side,
  onOpen,
}: {
  title: string;
  icon: ReactNode;
  side: "left" | "right";
  onOpen: () => void;
}) {
  return (
    <div className="rail">
      <button
        onClick={onOpen}
        className="icon-btn"
        title={`Show ${title}`}
        aria-label={`Show ${title}`}
      >
        {side === "left" ? (
          <PanelLeftOpen className="h-4 w-4" />
        ) : (
          <PanelRightOpen className="h-4 w-4" />
        )}
      </button>
      <span className="mt-1 text-ink-dim">{icon}</span>
      {/* Vertical label, so a collapsed panel still says what it holds. */}
      <span
        className="mt-2 whitespace-nowrap text-[0.7rem] font-semibold tracking-wide text-ink-dim"
        style={{ writingMode: "vertical-rl" }}
      >
        {title}
      </span>
    </div>
  );
}

/** Three panels need roughly 1024px before they stop fighting each other.
 *  Below that only the canvas is laid out, and a side panel opens over it. */
function useNarrow() {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 1023px)");
    const sync = () => setNarrow(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);
  return narrow;
}

export function Workspace({
  left,
  right,
  children,
}: {
  left: Side;
  right: Side;
  children: ReactNode;
}) {
  const [leftOpen, setLeftOpen] = useRemembered("ws.left", true);
  const [rightOpen, setRightOpen] = useRemembered("ws.right", true);
  const narrow = useNarrow();

  // On a narrow viewport a panel floats over the canvas instead of taking
  // width from it, and only one may be open at a time.
  if (narrow) {
    const open = rightOpen ? "right" : leftOpen ? "left" : null;
    const side = open === "right" ? right : open === "left" ? left : null;
    const close = () => {
      setLeftOpen(false);
      setRightOpen(false);
    };
    return (
      <div className="workspace relative">
        <Rail
          title={left.title}
          icon={left.icon}
          side="left"
          onOpen={() => {
            setRightOpen(false);
            setLeftOpen(true);
          }}
        />
        <section className="panel min-w-0 flex-1">{children}</section>
        <Rail
          title={right.title}
          icon={right.icon}
          side="right"
          onOpen={() => {
            setLeftOpen(false);
            setRightOpen(true);
          }}
        />

        {side && (
          <>
            <button
              onClick={close}
              aria-label="Close panel"
              className="absolute inset-0 z-20 bg-black/40 backdrop-blur-[1px]"
            />
            <section
              className={`panel absolute inset-y-2 z-30 w-[min(360px,82vw)] ${
                open === "left" ? "left-14" : "right-14"
              }`}
              style={{ boxShadow: "var(--shadow-pop)" }}
            >
              <div className="panel-head">
                <span className="panel-title">
                  <span className="text-ink-dim">{side.icon}</span>
                  {side.title}
                </span>
                <div className="flex items-center gap-1">
                  {side.actions}
                  <button onClick={close} className="icon-btn" title="Close" aria-label="Close">
                    {open === "left" ? (
                      <PanelLeftClose className="h-4 w-4" />
                    ) : (
                      <PanelRightClose className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
              <div className="panel-body">{side.children}</div>
            </section>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="workspace">
      {leftOpen ? (
        <section className="panel w-[clamp(240px,22vw,340px)] flex-shrink-0">
          <div className="panel-head">
            <span className="panel-title">
              <span className="text-ink-dim">{left.icon}</span>
              {left.title}
            </span>
            <div className="flex items-center gap-1">
              {left.actions}
              <button
                onClick={() => setLeftOpen(false)}
                className="icon-btn"
                title="Hide panel"
                aria-label="Hide panel"
              >
                <PanelLeftClose className="h-4 w-4" />
              </button>
            </div>
          </div>
          <div className="panel-body">{left.children}</div>
        </section>
      ) : (
        <Rail title={left.title} icon={left.icon} side="left" onOpen={() => setLeftOpen(true)} />
      )}

      {/* The canvas takes whatever the two side panels leave. */}
      <section className="panel min-w-0 flex-1">{children}</section>

      {rightOpen ? (
        <section className="panel w-[clamp(260px,24vw,380px)] flex-shrink-0">
          <div className="panel-head">
            <span className="panel-title">
              <span className="text-ink-dim">{right.icon}</span>
              {right.title}
            </span>
            <div className="flex items-center gap-1">
              {right.actions}
              <button
                onClick={() => setRightOpen(false)}
                className="icon-btn"
                title="Hide panel"
                aria-label="Hide panel"
              >
                <PanelRightClose className="h-4 w-4" />
              </button>
            </div>
          </div>
          <div className="panel-body">{right.children}</div>
        </section>
      ) : (
        <Rail
          title={right.title}
          icon={right.icon}
          side="right"
          onOpen={() => setRightOpen(true)}
        />
      )}
    </div>
  );
}
