/**
 * The mark: a play triangle built out of three lines of text.
 *
 * The whole product is one movement — a video becomes something you can read
 * — so the mark is that movement rather than a camera or a sparkle. Three
 * bars of falling width read as a paragraph at a glance and as a play button
 * at 16px, which is the size it spends most of its life at.
 *
 * It is drawn, not imported: an inline SVG inherits `currentColor`, so the
 * same file works on the light theme, the dark theme and a favicon without
 * three copies drifting apart.
 */
export function LogoMark({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" aria-hidden="true">
      <rect x="2" y="2" width="20" height="20" rx="6" fill="url(#vs-logo-grad)" />
      {/* Three bars, each shorter than the last, their right ends falling on
          the diagonal of a play triangle. */}
      <rect x="7" y="8" width="10" height="2.1" rx="1.05" fill="#fff" />
      <rect x="7" y="11.95" width="6.6" height="2.1" rx="1.05" fill="#fff" opacity="0.92" />
      <rect x="7" y="15.9" width="3.2" height="2.1" rx="1.05" fill="#fff" opacity="0.84" />
      <defs>
        <linearGradient id="vs-logo-grad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
          <stop stopColor="var(--c-accent-bright, #818cf8)" />
          <stop offset="1" stopColor="var(--c-accent, #6366f1)" />
        </linearGradient>
      </defs>
    </svg>
  );
}

/** Mark plus wordmark, for the header and the landing page. */
export function Logo({
  className = "",
  markClass = "h-[30px] w-[30px]",
  showWord = true,
}: {
  className?: string;
  markClass?: string;
  showWord?: boolean;
}) {
  return (
    <span className={`flex items-center gap-2.5 ${className}`}>
      <LogoMark className={markClass} />
      {showWord && (
        <span className="text-[0.95rem] font-bold tracking-[-0.01em] text-ink">
          Video<span className="text-accent">Sense</span>
        </span>
      )}
    </span>
  );
}
