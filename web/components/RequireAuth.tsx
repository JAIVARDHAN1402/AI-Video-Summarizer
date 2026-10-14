"use client";

import { LogIn, Sparkles } from "lucide-react";
import Link from "next/link";
import { openSignIn, useSession } from "@/components/Session";

/**
 * Keeps the working parts of the app behind sign-in.
 *
 * It asks rather than redirects. A redirect to the landing page loses the URL
 * somebody followed — a shared link to a specific video, say — and they have
 * no way back to it after signing in. This leaves the page where it is, so
 * signing in reveals what they came for.
 */
export function RequireAuth({
  children,
  what = "this page",
}: {
  children: React.ReactNode;
  what?: string;
}) {
  const { signedIn, ready } = useSession();

  // Nothing is drawn until the token has been read, because guessing wrong
  // means flashing a sign-in wall at somebody who is already signed in.
  if (!ready) {
    return (
      <div className="grid h-full place-items-center p-10">
        <div className="skeleton h-4 w-32" />
      </div>
    );
  }

  if (signedIn) return <>{children}</>;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid min-h-full max-w-[520px] place-items-center px-5 py-16 text-center">
        <div>
          <span className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-xl bg-accent/[0.12]">
            <Sparkles className="h-5 w-5 text-accent" />
          </span>
          <h1 className="text-[1.5rem] font-extrabold tracking-[-0.02em] text-ink">
            Sign in to open {what}
          </h1>
          <p className="mx-auto mt-2.5 max-w-[420px] text-[0.92rem] leading-relaxed text-ink-muted">
            Processing a video, your library and your notebooks are tied to an
            account, because the allowance that pays for them is. Signing in
            takes a code sent to your email — there is no password to make up.
          </p>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-2.5">
            <button onClick={openSignIn} className="btn btn-primary">
              <LogIn className="h-4 w-4" />
              Sign in
            </button>
            <Link href="/" className="btn no-underline">
              What is this?
            </Link>
          </div>

          <p className="mt-5 text-[0.78rem] leading-relaxed text-ink-dim">
            The free plan covers videos up to 4 hours, with no card and nothing
            to cancel.
          </p>
        </div>
      </div>
    </div>
  );
}
