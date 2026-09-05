import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#07080B",
        "bg-elevated": "#0A0C11",
        surface: "#0E1015",
        "surface-raised": "#141821",
        "surface-hover": "#1A1F2A",
        "surface-active": "#202634",
        border: "rgba(255,255,255,0.06)",
        "border-strong": "rgba(255,255,255,0.11)",
        ink: "#EDEFF5",
        "ink-muted": "#949CB0",
        "ink-dim": "#616A7D",
        accent: "#6D6AF8",
        "accent-bright": "#8B8AFB",
        violet: "#A855F7",
        cyan: "#22D3EE",
        success: "#34D399",
        warning: "#FBBF24",
        danger: "#F87171",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: { xs: "6px", sm: "9px", DEFAULT: "13px", lg: "18px" },
    },
  },
  plugins: [],
} satisfies Config;
