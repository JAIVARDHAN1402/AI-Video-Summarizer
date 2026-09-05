import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Video Summarizer",
  description:
    "Understand any video without watching it — a teachable explanation, chapters, and cited Q&A.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="mx-auto max-w-[1200px] px-5 pb-24 pt-6">
          <header className="mb-9 flex items-center justify-between border-b border-border pb-4">
            <Link href="/" className="flex items-center gap-3 no-underline">
              <span className="flex h-9 w-9 items-center justify-center rounded-[9px] bg-accent shadow-[0_2px_8px_rgba(79,70,229,0.28)]">
                <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none">
                  <path d="M9 7.5v9l7-4.5-7-4.5z" fill="#fff" />
                </svg>
              </span>
              <div className="leading-tight">
                <p className="m-0 text-[1.02rem] font-bold tracking-tight text-ink">
                  AI Video Summarizer
                </p>
                <p className="m-0 text-[0.75rem] text-ink-dim">
                  Understand any video without watching it
                </p>
              </div>
            </Link>
            <nav className="flex items-center gap-1 text-[0.85rem] font-medium text-ink-muted">
              <Link href="/" className="rounded-sm px-2.5 py-1.5 transition-colors hover:bg-surface-hover hover:text-ink">
                New
              </Link>
              <Link href="/library" className="rounded-sm px-2.5 py-1.5 transition-colors hover:bg-surface-hover hover:text-ink">
                Library
              </Link>
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
