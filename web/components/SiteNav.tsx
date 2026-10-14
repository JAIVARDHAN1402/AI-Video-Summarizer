"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { AccountMenu } from "@/components/AccountMenu";
import { Logo } from "@/components/Logo";
import { useSession } from "@/components/Session";
import { ThemeToggle } from "@/components/ThemeToggle";

const LINKS = [
  { href: "/app", label: "New" },
  { href: "/library", label: "Library" },
  { href: "/notebooks", label: "Notebooks" },
];

/**
 * The header, which is two headers.
 *
 * Signed out there is nothing to navigate to — every one of those links
 * leads to a sign-in wall — so offering them is just three ways to be told
 * no. The bar carries the mark and a way in instead. Signed in it becomes
 * the working navigation.
 */
export function SiteNav() {
  const { signedIn, ready } = useSession();
  const pathname = usePathname();

  return (
    <header className="topbar">
      {/* The mark goes to the landing page when signed out and to the work
          when signed in, which is what clicking a logo is expected to do. */}
      <Link href={signedIn ? "/app" : "/"} className="no-underline">
        <Logo />
      </Link>

      <nav className="flex items-center gap-1 text-[0.85rem] font-medium text-ink-muted">
        {/* Held back until the token has been read: rendering the links and
            removing them a frame later reads as a glitch. */}
        {ready && signedIn &&
          LINKS.map((link) => {
            const active =
              pathname === link.href || pathname.startsWith(`${link.href}/`);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-md px-3 py-1.5 no-underline transition-colors hover:bg-surface-hover hover:text-ink ${
                  active ? "bg-accent/[0.10] text-accent" : ""
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        {ready && !signedIn && pathname !== "/" && (
          <Link
            href="/"
            className="rounded-md px-3 py-1.5 no-underline transition-colors hover:bg-surface-hover hover:text-ink"
          >
            What is this?
          </Link>
        )}
        <span className="mx-1 h-5 w-px bg-border" />
        <ThemeToggle />
        <AccountMenu />
      </nav>
    </header>
  );
}
