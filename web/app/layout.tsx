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
          <header className="mb-8 flex items-center justify-between border-b border-border pb-4">
            <Link href="/" className="flex items-center gap-3 no-underline">
              <span className="flex h-10 w-10 items-center justify-center rounded-[11px] bg-gradient-to-br from-accent to-violet shadow-[0_6px_20px_rgba(109,106,248,0.38)]">
                <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none">
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
            <nav className="flex items-center gap-4 text-[0.85rem] text-ink-muted">
              <Link href="/" className="hover:text-ink">New</Link>
              <Link href="/library" className="hover:text-ink">Library</Link>
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
