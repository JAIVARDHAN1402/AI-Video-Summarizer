import type { Config } from "tailwindcss";

// Every colour is an RGB channel triple in a CSS variable so the same utility
// classes work in both themes and opacity modifiers (bg-accent/[0.08]) still
// compose. The values live in app/globals.css under :root and .dark.
const c = (v: string) => `rgb(var(${v}) / <alpha-value>)`;

export default {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: c("--c-bg"),
        "bg-elevated": c("--c-bg-elevated"),
        surface: c("--c-surface"),
        "surface-raised": c("--c-surface-raised"),
        "surface-hover": c("--c-surface-hover"),
        "surface-active": c("--c-surface-active"),
        border: c("--c-border"),
        "border-strong": c("--c-border-strong"),
        ink: c("--c-ink"),
        "ink-muted": c("--c-ink-muted"),
        "ink-dim": c("--c-ink-dim"),
        accent: c("--c-accent"),
        "accent-bright": c("--c-accent-bright"),
        violet: c("--c-violet"),
        cyan: c("--c-cyan"),
        success: c("--c-success"),
        warning: c("--c-warning"),
        danger: c("--c-danger"),
        "code-bg": c("--c-code-bg"),
        "code-ink": c("--c-code-ink"),
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: { xs: "5px", sm: "7px", DEFAULT: "10px", lg: "14px" },
      boxShadow: {
        card: "var(--shadow-card)",
        "card-hover": "var(--shadow-card-hover)",
        pop: "var(--shadow-pop)",
      },
    },
  },
  plugins: [],
} satisfies Config;
