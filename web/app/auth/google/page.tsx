"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { api, setSessionToken } from "@/lib/api";

/** Completes the browser half of the Google OAuth redirect. */
function GoogleAuthCallbackContent() {
  const router = useRouter();
  const params = useSearchParams();
  const [message, setMessage] = useState("Finishing Google sign-in…");

  useEffect(() => {
    const code = params.get("code");
    const state = params.get("state");
    const expected = sessionStorage.getItem("vs-google-state");
    if (!code) {
      setMessage("Google did not return a sign-in code.");
      return;
    }
    if (!state || !expected || state !== expected) {
      setMessage("This sign-in link is no longer valid. Start again from the account menu.");
      return;
    }
    sessionStorage.removeItem("vs-google-state");
    api.googleCallback(code)
      .then((session) => {
        setSessionToken(session.token);
        router.replace("/app");
      })
      .catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : "Could not finish Google sign-in.");
      });
  }, [params, router]);

  return (
    <div className="page-col">
      <section className="card max-w-[520px] p-6">
        <h1 className="m-0 text-xl font-bold text-ink">Google sign-in</h1>
        <p className="mt-3 text-[0.9rem] text-ink-muted">{message}</p>
        {message !== "Finishing Google sign-in…" && <Link href="/" className="btn mt-4 inline-flex">Back home</Link>}
      </section>
    </div>
  );
}

export default function GoogleAuthCallback() {
  return (
    <Suspense fallback={<div className="page-col text-ink-muted">Preparing sign-in…</div>}>
      <GoogleAuthCallbackContent />
    </Suspense>
  );
}
