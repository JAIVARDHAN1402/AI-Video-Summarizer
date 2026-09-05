import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Light-first, clean & bright. One accent (indigo), used sparingly.
        bg: "#FFFFFF",
        "bg-elevated": "#F7F8FA",
        surface: "#FFFFFF",
        "surface-raised": "#FBFBFC",
        "surface-hover": "#F3F4F6",
        "surface-active": "#EBEDF0",
        border: "#E8E9EC",
        "border-strong": "#D6D8DD",
        ink: "#18181B",
        "ink-muted": "#52525B",
        "ink-dim": "#8A8A94",
        accent: "#4F46E5",
        "accent-bright": "#4338CA",
        violet: "#7C3AED",
        cyan: "#0E7490",
        success: "#059669",
        warning: "#B45309",
        danger: "#DC2626",
        // A single dark tone for code islands on the light page.
        "code-bg": "#1B1B1F",
        "code-ink": "#E7E7EA",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: { xs: "5px", sm: "7px", DEFAULT: "10px", lg: "14px" },
      boxShadow: {
        card: "0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06)",
        "card-hover": "0 2px 10px rgba(16,24,40,0.09)",
        pop: "0 8px 28px rgba(16,24,40,0.12)",
      },
    },
  },
  plugins: [],
} satisfies Config;
