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
      {/* The shell owns the viewport height; each page decides whether its
          content is a scrolling column or a set of independently scrolling
          panels. Nothing below this point scrolls the window itself. */}
      <body className="app-shell">
        <header className="topbar">
          <Link href="/" className="flex items-center gap-2.5 no-underline">
            <span className="brand-mark">
              <svg viewBox="0 0 24 24" className="h-[17px] w-[17px]" fill="none">
                <path d="M9 7.5v9l7-4.5-7-4.5z" fill="#fff" />
              </svg>
            </span>
            <span className="hidden text-[0.95rem] font-bold tracking-tight text-ink sm:block">
              AI Video Summarizer
            </span>
          </Link>

          <nav className="flex items-center gap-1 text-[0.85rem] font-medium text-ink-muted">
            <Link
              href="/"
              className="rounded-md px-3 py-1.5 no-underline transition-colors hover:bg-surface-hover hover:text-ink"
            >
              New
            </Link>
            <Link
              href="/library"
              className="rounded-md px-3 py-1.5 no-underline transition-colors hover:bg-surface-hover hover:text-ink"
            >
              Library
            </Link>
            <Link
              href="/notebooks"
              className="rounded-md px-3 py-1.5 no-underline transition-colors hover:bg-surface-hover hover:text-ink"
            >
              Notebooks
            </Link>
            <span className="mx-1 h-5 w-px bg-border" />
            <ThemeToggle />
          </nav>
        </header>

        <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
      </body>
    </html>
  );
}
