"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

type Choice = "light" | "dark" | "system";

const OPTIONS: { value: Choice; label: string; Icon: typeof Sun }[] = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
  { value: "system", label: "System", Icon: Monitor },
];

function systemPrefersDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

/** Apply a choice to <html> and remember it. "system" clears the override. */
export function applyTheme(choice: Choice) {
  const dark = choice === "dark" || (choice === "system" && systemPrefersDark());
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.style.colorScheme = dark ? "dark" : "light";
  try {
    if (choice === "system") localStorage.removeItem("theme");
    else localStorage.setItem("theme", choice);
  } catch {
    /* private mode / storage blocked — the class still applied */
  }
}

export function ThemeToggle() {
  // Render the light icon until mounted so server and client markup agree;
  // the inline script in layout.tsx has already painted the right theme.
  const [choice, setChoice] = useState<Choice>("system");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem("theme");
    } catch {
      /* ignore */
    }
    setChoice(stored === "light" || stored === "dark" ? stored : "system");
    setMounted(true);
  }, []);

  // Follow the OS while the user is on "system".
  useEffect(() => {
    if (choice !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyTheme("system");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [choice]);

  function pick(next: Choice) {
    setChoice(next);
    applyTheme(next);
  }

  return (
    <div
      className="flex items-center gap-0.5 rounded-sm border border-border bg-bg-elevated p-0.5"
      role="group"
      aria-label="Colour theme"
    >
      {OPTIONS.map(({ value, label, Icon }) => {
        const active = mounted && choice === value;
        return (
          <button
            key={value}
            type="button"
            onClick={() => pick(value)}
            title={label}
            aria-label={label}
            aria-pressed={active}
            className={`rounded-xs p-1.5 transition-colors ${
              active
                ? "bg-surface text-ink shadow-card"
                : "text-ink-dim hover:text-ink"
            }`}
          >
            <Icon className="h-[15px] w-[15px]" strokeWidth={2} />
          </button>
        );
      })}
    </div>
  );
}
