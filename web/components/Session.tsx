"use client";

import { useEffect, useState } from "react";
import { hasSessionToken } from "@/lib/api";

/**
 * Whether somebody is signed in, in a form the UI can render against.
 *
 * `ready` exists because the token lives in localStorage, which the server
 * cannot see: the first render on the client has to assume nothing. Without
 * it every gated page flashes its signed-out state for a frame before the
 * token is read, which looks like being logged out at random.
 */
export function useSession(): { signedIn: boolean; ready: boolean } {
  const [state, setState] = useState<{ signedIn: boolean; ready: boolean }>({
    signedIn: false,
    ready: false,
  });

  useEffect(() => {
    const read = () => setState({ signedIn: hasSessionToken(), ready: true });
    read();
    // setSessionToken fires the first; the second is this app open in another
    // tab, where signing out should not leave this one looking signed in.
    window.addEventListener("vs-auth-change", read);
    window.addEventListener("storage", read);
    return () => {
      window.removeEventListener("vs-auth-change", read);
      window.removeEventListener("storage", read);
    };
  }, []);

  return state;
}

/** Open the account menu from anywhere — the one place sign-in lives. */
export function openSignIn(): void {
  window.dispatchEvent(new Event("vs-open-auth"));
}
