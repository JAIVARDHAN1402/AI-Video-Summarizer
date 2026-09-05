import type { Metadata } from "next";
import Link from "next/link";
import { ThemeToggle } from "@/components/ThemeToggle";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Video Summarizer",
  description:
    "Understand any video without watching it — a teachable explanation, chapters, and cited Q&A.",
};

// Runs before first paint so the stored theme is applied without a flash of
// the wrong one. Kept tiny and dependency-free on purpose.
const NO_FLASH = `(function(){try{
var s=localStorage.getItem('theme');
var d=s?s==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;
if(d)document.documentElement.classList.add('dark');
document.documentElement.style.colorScheme=d?'dark':'light';
}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: NO_FLASH }} />
      </head>
      <body>
        <div className="mx-auto max-w-[1200px] px-5 pb-24 pt-6">
          <header className="mb-9 flex items-center justify-between gap-4 border-b border-border pb-4">
            <Link href="/" className="flex items-center gap-3 no-underline">
              <span className="flex h-9 w-9 items-center justify-center rounded-[9px] bg-accent shadow-[0_2px_8px_rgb(var(--c-accent)/0.30)]">
                <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none">
                  <path d="M9 7.5v9l7-4.5-7-4.5z" fill="#fff" />
                </svg>
              </span>
              <div className="leading-tight">
                <p className="m-0 text-[1.02rem] font-bold tracking-tight text-ink">
                  AI Video Summarizer
                </p>
                <p className="m-0 hidden text-[0.75rem] text-ink-dim sm:block">
                  Understand any video without watching it
                </p>
              </div>
            </Link>
            <nav className="flex items-center gap-2 text-[0.85rem] font-medium text-ink-muted">
              <Link
                href="/"
                className="rounded-sm px-2.5 py-1.5 transition-colors hover:bg-surface-hover hover:text-ink"
              >
                New
              </Link>
              <Link
                href="/library"
                className="rounded-sm px-2.5 py-1.5 transition-colors hover:bg-surface-hover hover:text-ink"
              >
                Library
              </Link>
              <ThemeToggle />
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
