import type { Metadata } from "next";
import { SiteNav } from "@/components/SiteNav";
import "./globals.css";

export const metadata: Metadata = {
  title: "VideoSense — understand any video without watching it",
  description:
    "A teachable explanation of any video, section by section, with chapters, " +
    "a practice set for coding videos, and answers that cite the moment they came from.",
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
        <SiteNav />
        <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
      </body>
    </html>
  );
}
