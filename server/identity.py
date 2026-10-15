"""Who the request is from: one-time codes, Google, and the session token.

Three ways in, one account. The account is keyed on the email address or the
phone number itself rather than on a provider's user id, so signing in with
Google and later with a code sent to the same inbox lands on the same
account instead of quietly creating a second one.

No password is stored anywhere, on purpose. Every route in is either a code
that expires in minutes or Google's own sign-in, so there is nothing in this
database worth stealing for use elsewhere.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import logging
import re
import secrets
from typing import Any, Optional

from server import config, db, notify

logger = logging.getLogger("server.identity")

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
_PHONE = re.compile(r"^\+?[1-9]\d{7,14}$")


class IdentityError(Exception):
    """Something the caller got wrong, safe to show them."""


# --- destinations ---------------------------------------------------------

def normalise(channel: str, destination: str) -> str:
    """One spelling per person, so two sign-ins cannot make two accounts."""
    value = (destination or "").strip()
    if channel == "email":
        value = value.lower()
        if not _EMAIL.match(value):
            raise IdentityError("that does not look like an email address")
        return value
    if channel == "phone":
        value = re.sub(r"[\s\-()]", "", value)
        if value and not value.startswith("+"):
            # Bare ten-digit numbers are the common case here, and storing
            # them unqualified would collide with the same digits elsewhere.
            value = ("+91" + value) if len(value) == 10 else ("+" + value)
        if not _PHONE.match(value):
            raise IdentityError("that does not look like a phone number")
        return value
    raise IdentityError(f"unknown channel {channel!r}")


# --- one-time codes -------------------------------------------------------

def _hash(code: str, destination: str) -> str:
    """Salted with the destination, so one rainbow table is not enough."""
    return hashlib.sha256(f"{destination}:{code}".encode()).hexdigest()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def request_code(session, channel: str, destination: str) -> dict[str, Any]:
    """Make a code, store its hash, and send it."""
    destination = normalise(channel, destination)

    # How recently anything was *sent* here, whatever became of it. Asking
    # only about live codes left a hole: six wrong guesses consume a code, and
    # the window then looks clear, so a script could alternate between
    # guessing and resending and mail a stranger's inbox without limit. What
    # is being rate-limited is the sending, not the code.
    last_sent = (
        session.query(db.OtpCode)
        .filter(db.OtpCode.destination == destination)
        .order_by(db.OtpCode.created_at.desc())
        .first()
    )
    if last_sent is not None:
        age = (_now() - db._aware(last_sent.created_at)).total_seconds()
        if age < config.OTP_RESEND_SEC:
            raise IdentityError(
                f"a code was just sent; try again in "
                f"{int(config.OTP_RESEND_SEC - age)} seconds"
            )

    # A new code supersedes any live one rather than living beside it, so two
    # codes are never valid for the same inbox at once.
    for live in (
        session.query(db.OtpCode)
        .filter(db.OtpCode.destination == destination, db.OtpCode.consumed.is_(False))
        .all()
    ):
        live.consumed = True

    code = "".join(secrets.choice("0123456789") for _ in range(config.OTP_LENGTH))
    row = db.OtpCode(
        destination=destination,
        channel=channel,
        code_hash=_hash(code, destination),
        expires_at=_now() + dt.timedelta(seconds=config.OTP_TTL_SEC),
    )
    session.add(row)
    session.commit()

    try:
        notify.send_code(channel, destination, code)
    except notify.DeliveryError as exc:
        row.consumed = True       # nobody can use a code that was never sent
        session.commit()
        raise IdentityError(str(exc)) from exc

    return {"sent": True, "channel": channel, "destination": destination,
            "expires_in_sec": config.OTP_TTL_SEC}


def verify_code(session, channel: str, destination: str, code: str) -> db.User:
    """Check a code and return the account it belongs to, creating it if new."""
    destination = normalise(channel, destination)
    row = (
        session.query(db.OtpCode)
        .filter(db.OtpCode.destination == destination, db.OtpCode.consumed.is_(False))
        .order_by(db.OtpCode.created_at.desc())
        .first()
    )
    if row is None:
        raise IdentityError("ask for a code first")
    if db._aware(row.expires_at) < _now():
        row.consumed = True
        session.commit()
        raise IdentityError("that code has expired; ask for a new one")
    if row.attempts >= config.OTP_MAX_ATTEMPTS:
        row.consumed = True
        session.commit()
        raise IdentityError("too many wrong tries; ask for a new code")

    # compare_digest so a wrong code cannot be narrowed down by timing.
    if not hmac.compare_digest(row.code_hash, _hash((code or "").strip(), destination)):
        row.attempts += 1
        session.commit()
        left = max(config.OTP_MAX_ATTEMPTS - row.attempts, 0)
        raise IdentityError(f"that code is not right ({left} tries left)")

    row.consumed = True
    session.commit()
    return _upsert(session, channel=channel, destination=destination, method="otp")


# --- Google ---------------------------------------------------------------

def google_configured() -> bool:
    return bool(config.GOOGLE_CLIENT_ID and config.GOOGLE_CLIENT_SECRET)


def google_auth_url(state: str) -> str:
    if not google_configured():
        raise IdentityError("Google sign-in is not configured on this server")
    from urllib.parse import urlencode

    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": config.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account",
        "state": state,
    })


def google_sign_in(session, code: str) -> db.User:
    """Exchange the one-time code Google sent back for an account."""
    if not google_configured():
        raise IdentityError("Google sign-in is not configured on this server")

    import requests

    token = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": config.GOOGLE_CLIENT_ID,
            "client_secret": config.GOOGLE_CLIENT_SECRET,
            "redirect_uri": config.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    if token.status_code >= 300:
        raise IdentityError(f"Google refused the sign-in: {token.text[:200]}")

    access_token = token.json().get("access_token")
    if not access_token:
        raise IdentityError("Google did not return an access token")

    profile = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    if profile.status_code >= 300:
        raise IdentityError(f"could not read the Google profile: {profile.text[:200]}")
    data = profile.json()

    if not data.get("email_verified"):
        # An unverified address would let anyone claim an account by putting
        # someone else's email on a Google profile.
        raise IdentityError("that Google account's email address is not verified")

    return _upsert(
        session, channel="email", destination=(data.get("email") or "").lower(),
        method="google", name=data.get("name"), picture=data.get("picture"),
    )


# --- accounts -------------------------------------------------------------

def _upsert(session, *, channel: str, destination: str, method: str,
            name: Optional[str] = None, picture: Optional[str] = None) -> db.User:
    field = db.User.email if channel == "email" else db.User.phone
    user = session.query(db.User).filter(field == destination).first()
    if user is None:
        user = db.User(signup_method=method)
        setattr(user, "email" if channel == "email" else "phone", destination)
        session.add(user)
    if name and not user.name:
        user.name = name
    if picture:
        user.picture = picture
    user.last_seen_at = _now()
    session.commit()
    return user


# --- session token --------------------------------------------------------

def issue_token(user: db.User) -> str:
    """A signed token the browser keeps; the server stores no session."""
    if not config.JWT_SECRET:
        raise IdentityError(
            "VS_JWT_SECRET is not set, so sign-in tokens cannot be signed"
        )
    import jwt

    now = _now()
    claims = {
        "sub": user.id,
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(days=config.SESSION_TTL_DAYS)).timestamp()),
        "email": user.email,
        "tier": user.effective_tier.value,
    }
    if config.JWT_AUDIENCE:
        claims["aud"] = config.JWT_AUDIENCE
    return jwt.encode(claims, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def user_from_principal(session, principal: dict) -> Optional[db.User]:
    """The account a verified token belongs to, or None for anything else.

    A static API key is a machine, not a person: it has no account and so no
    allowance, which is why this returns None rather than inventing one.
    """
    if not principal or principal.get("kind") != "jwt":
        return None
    sub = principal.get("sub")
    if not sub:
        return None
    return session.query(db.User).filter(db.User.id == sub).first()
