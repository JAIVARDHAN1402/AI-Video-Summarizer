"use client";

import { CreditCard, LogIn, LogOut, Mail, Phone, UserRound } from "lucide-react";
import { useEffect, useState } from "react";
import {
  api,
  hasSessionToken,
  setSessionToken,
  type Account,
  type AuthMethods,
  type BillingPlan,
  type Quota,
} from "@/lib/api";

declare global {
  interface Window {
    Razorpay?: new (options: Record<string, unknown>) => { open: () => void };
  }
}

type Channel = "email" | "phone";

function messageOf(error: unknown, fallback: string) {
  return error instanceof Error && error.message ? error.message : fallback;
}

function googleState() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function loadRazorpay(): Promise<void> {
  if (window.Razorpay) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const existing = document.getElementById("razorpay-checkout") as HTMLScriptElement | null;
    if (existing) {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error("Could not load Razorpay Checkout.")), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.id = "razorpay-checkout";
    script.src = "https://checkout.razorpay.com/v1/checkout.js";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Could not load Razorpay Checkout."));
    document.head.appendChild(script);
  });
}

/** Sign-in, allowance, and premium upgrade controls for the browser shell. */
export function AccountMenu() {
  const [methods, setMethods] = useState<AuthMethods | null>(null);
  const [account, setAccount] = useState<Account | null>(null);
  const [quota, setQuota] = useState<Quota | null>(null);
  const [plan, setPlan] = useState<BillingPlan | null>(null);
  const [open, setOpen] = useState(false);
  const [channel, setChannel] = useState<Channel>("email");
  const [destination, setDestination] = useState("");
  const [code, setCode] = useState("");
  const [codeSent, setCodeSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  async function refresh() {
    try {
      const available = await api.authMethods();
      setMethods(available);
      if (!available.email && available.phone) setChannel("phone");
      const billing = await api.billingPlan();
      setPlan(billing);
      if (hasSessionToken() || !available.auth_required) {
        const me = await api.me();
        setAccount(me.user);
        setQuota(me.quota);
      } else {
        setAccount(null);
        setQuota(null);
      }
    } catch (err) {
      setError(messageOf(err, "Could not reach the account service."));
    }
  }

  useEffect(() => {
    void refresh();
    const changed = () => void refresh();
    const openSignIn = () => {
      setOpen(true);
      void refresh();
    };
    window.addEventListener("vs-auth-change", changed);
    window.addEventListener("vs-open-auth", openSignIn);
    return () => {
      window.removeEventListener("vs-auth-change", changed);
      window.removeEventListener("vs-open-auth", openSignIn);
    };
  }, []);

  async function requestCode() {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await api.requestOtp(channel, destination);
      setDestination(result.destination);
      setCodeSent(true);
      setNotice(
        methods?.dev_echo
          ? "Development mode: the sign-in code is in the API server log."
          : `A code was sent to ${result.destination}.`,
      );
    } catch (err) {
      setError(messageOf(err, "Could not send a sign-in code."));
    } finally {
      setBusy(false);
    }
  }

  async function verifyCode() {
    setBusy(true);
    setError("");
    try {
      const session = await api.verifyOtp(channel, destination, code);
      setSessionToken(session.token);
      setAccount(session.user);
      setQuota(session.quota);
      setNotice("Signed in successfully.");
      setCode("");
      setCodeSent(false);
    } catch (err) {
      setError(messageOf(err, "That code could not be verified."));
    } finally {
      setBusy(false);
    }
  }

  async function signInWithGoogle() {
    setBusy(true);
    setError("");
    try {
      const state = googleState();
      sessionStorage.setItem("vs-google-state", state);
      const { url } = await api.googleUrl(state);
      window.location.assign(url);
    } catch (err) {
      setError(messageOf(err, "Could not start Google sign-in."));
      setBusy(false);
    }
  }

  async function upgrade() {
    if (!plan?.configured) return;
    setBusy(true);
    setError("");
    try {
      const order = await api.billingOrder();
      await loadRazorpay();
      if (!window.Razorpay) throw new Error("Razorpay Checkout did not load.");
      const checkout = new window.Razorpay({
        key: order.key_id,
        amount: order.amount_paise,
        currency: order.currency,
        name: "AI Video Summarizer",
        description: "Premium access",
        order_id: order.order_id,
        handler: async (payment: { razorpay_payment_id: string; razorpay_signature: string }) => {
          try {
            const result = await api.billingConfirm(
              order.order_id,
              payment.razorpay_payment_id,
              payment.razorpay_signature,
            );
            setAccount(result.user);
            setQuota(result.quota);
            setNotice("Premium is active. Thank you!");
          } catch (err) {
            setError(messageOf(err, "Payment was received but could not be verified."));
          }
        },
        theme: { color: "#6366f1" },
      });
      checkout.open();
    } catch (err) {
      setError(messageOf(err, "Could not open the payment form."));
    } finally {
      setBusy(false);
    }
  }

  function signOut() {
    setSessionToken(null);
    setAccount(null);
    setQuota(null);
    setCodeSent(false);
    setNotice("Signed out on this browser.");
  }

  // An open local API has no account requirement. Avoid adding a sign-in
  // button that cannot mint a token because VS_JWT_SECRET is intentionally off.
  if (methods && !methods.auth_required && !hasSessionToken()) return null;

  return (
    <div className="relative">
      <button
        onClick={() => { setOpen((value) => !value); setError(""); }}
        className="btn gap-1.5 px-3 py-1.5 text-[0.8rem]"
        aria-expanded={open}
      >
        <UserRound className="h-3.5 w-3.5" />
        {account ? account.name || account.email || account.phone || "Account" : "Sign in"}
      </button>

      {open && (
        <section className="absolute right-0 top-[calc(100%+0.5rem)] z-50 w-[min(92vw,360px)] rounded-xl border border-border bg-surface p-4 shadow-card-hover">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div>
              <h2 className="m-0 text-[0.95rem] font-semibold text-ink">
                {account ? "Your account" : "Sign in"}
              </h2>
              <p className="mt-1 text-[0.78rem] leading-relaxed text-ink-dim">
                {account ? "Your allowance applies to processing requests." : "Sign in to use your personal allowance."}
              </p>
            </div>
            <button className="icon-btn" onClick={() => setOpen(false)} aria-label="Close account menu">×</button>
          </div>

          {account ? (
            <div className="space-y-3">
              <div className="rounded-lg border border-border bg-surface-raised p-3 text-[0.82rem]">
                <div className="flex justify-between gap-3"><span className="text-ink-dim">Plan</span><strong className="capitalize text-ink">{quota?.tier || account.tier}</strong></div>
                {quota && <div className="mt-1 flex justify-between gap-3"><span className="text-ink-dim">Today</span><span className="text-ink">{quota.remaining} of {quota.daily_calls} calls left</span></div>}
                {quota && <div className="mt-1 flex justify-between gap-3"><span className="text-ink-dim">Video limit</span><span className="text-ink">{quota.max_video_hours} hours</span></div>}
              </div>
              {plan?.configured && quota?.tier !== "premium" && (
                <button onClick={() => void upgrade()} disabled={busy} className="btn btn-primary w-full gap-2">
                  <CreditCard className="h-4 w-4" /> Upgrade for ₹{(plan.amount_paise / 100).toFixed(0)} / {plan.days} days
                </button>
              )}
              <button onClick={signOut} className="btn w-full gap-2"><LogOut className="h-4 w-4" /> Sign out</button>
            </div>
          ) : (
            <div className="space-y-3">
              {!methods?.google && !methods?.email && !methods?.phone && (
                <p className="rounded-md bg-warning/[0.10] px-2.5 py-2 text-[0.78rem] leading-relaxed text-warning">
                  This server is protected with static API credentials. Configure a JWT secret and a sign-in method to use browser accounts.
                </p>
              )}
              {methods?.google && (
                <button onClick={() => void signInWithGoogle()} disabled={busy} className="btn w-full gap-2">
                  <LogIn className="h-4 w-4" /> Continue with Google
                </button>
              )}
              {(methods?.email || methods?.phone) && <div className="flex rounded-md border border-border p-0.5 text-[0.78rem]">
                {methods?.email && <button onClick={() => { setChannel("email"); setCodeSent(false); }} className={`flex-1 rounded px-2 py-1 ${channel === "email" ? "bg-surface-raised text-ink" : "text-ink-dim"}`}><Mail className="mr-1 inline h-3 w-3" />Email</button>}
                {methods?.phone && <button onClick={() => { setChannel("phone"); setCodeSent(false); }} className={`flex-1 rounded px-2 py-1 ${channel === "phone" ? "bg-surface-raised text-ink" : "text-ink-dim"}`}><Phone className="mr-1 inline h-3 w-3" />Phone</button>}
              </div>}
              {(methods?.email || methods?.phone) && <>
                <input className="input" value={destination} onChange={(e) => setDestination(e.target.value)} placeholder={channel === "email" ? "you@example.com" : "+91 98765 43210"} inputMode={channel === "phone" ? "tel" : "email"} />
                {codeSent && <input className="input" value={code} onChange={(e) => setCode(e.target.value)} placeholder="Six-digit code" inputMode="numeric" maxLength={12} />}
                <button onClick={() => void (codeSent ? verifyCode() : requestCode())} disabled={busy || !destination.trim() || (codeSent && !code.trim())} className="btn btn-primary w-full">
                  {busy ? "Please wait…" : codeSent ? "Verify and sign in" : "Send sign-in code"}
                </button>
              </>}
            </div>
          )}
          {notice && <p className="mt-3 rounded-md bg-success/[0.10] px-2.5 py-2 text-[0.78rem] text-success">{notice}</p>}
          {error && <p className="mt-3 rounded-md bg-danger/[0.10] px-2.5 py-2 text-[0.78rem] text-danger">{error}</p>}
        </section>
      )}
    </div>
  );
}
