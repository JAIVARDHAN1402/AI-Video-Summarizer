"""Upgrading to premium, through Razorpay.

Test mode by default and deliberately so: live keys need a registered
business, GST registration and a refund policy, none of which a project needs
in order to demonstrate the flow. The difference between the two is which
keys are configured — the code is identical, which is the point.

The signature check is the part that matters. A client that could report its
own success would let anybody upgrade themselves by calling the endpoint, so
the only thing trusted here is an HMAC the server computes itself over the
order and payment ids using the key secret.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import logging
from typing import Any

from server import config, db

logger = logging.getLogger("server.billing")


class BillingError(Exception):
    """Something the caller can be told about."""


def configured() -> bool:
    return bool(config.RAZORPAY_KEY_ID and config.RAZORPAY_KEY_SECRET)


def plan() -> dict[str, Any]:
    """What is on offer, for the upgrade screen."""
    from server import quota

    free = quota.TIERS[db.Tier.FREE]
    premium = quota.TIERS[db.Tier.PREMIUM]
    return {
        "configured": configured(),
        "mode": config.RAZORPAY_MODE,
        "key_id": config.RAZORPAY_KEY_ID,
        "amount_paise": config.PREMIUM_PRICE_PAISE,
        "currency": "INR",
        "days": config.PREMIUM_DAYS,
        "free": {"max_video_hours": free.max_video_hours,
                 "daily_calls": free.daily_calls},
        "premium": {"max_video_hours": premium.max_video_hours,
                    "daily_calls": premium.daily_calls},
    }


def create_order(session, user: db.User) -> dict[str, Any]:
    """Open a Razorpay order for this user's upgrade."""
    if not configured():
        raise BillingError("payments are not configured on this server")

    import requests

    response = requests.post(
        "https://api.razorpay.com/v1/orders",
        auth=(config.RAZORPAY_KEY_ID, config.RAZORPAY_KEY_SECRET),
        json={
            "amount": config.PREMIUM_PRICE_PAISE,
            "currency": "INR",
            "notes": {"user_id": user.id, "plan": "premium"},
        },
        timeout=20,
    )
    if response.status_code >= 300:
        raise BillingError(f"Razorpay refused the order: {response.text[:200]}")

    order = response.json()
    session.add(db.Payment(
        user_id=user.id, order_id=order["id"],
        amount_paise=int(order.get("amount") or config.PREMIUM_PRICE_PAISE),
        currency=order.get("currency") or "INR",
        status="created", mode=config.RAZORPAY_MODE,
    ))
    session.commit()

    return {
        "order_id": order["id"],
        "amount_paise": order.get("amount"),
        "currency": order.get("currency"),
        "key_id": config.RAZORPAY_KEY_ID,
        "mode": config.RAZORPAY_MODE,
    }


def _signature_ok(order_id: str, payment_id: str, signature: str) -> bool:
    expected = hmac.new(
        config.RAZORPAY_KEY_SECRET.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, (signature or "").strip())


def confirm(session, user: db.User, order_id: str, payment_id: str,
            signature: str) -> db.User:
    """Verify the payment and move the account to premium.

    Nothing the client says about whether it paid is trusted: the only
    evidence accepted is a signature this server can recompute.
    """
    if not configured():
        raise BillingError("payments are not configured on this server")

    row = (
        session.query(db.Payment)
        .filter(db.Payment.order_id == order_id, db.Payment.user_id == user.id)
        .first()
    )
    if row is None:
        raise BillingError("that order does not belong to this account")
    if row.status == "paid":
        return user          # a repeated callback must not extend the plan twice

    if not _signature_ok(order_id, payment_id, signature):
        row.status = "failed"
        session.commit()
        logger.warning("rejected a payment with a bad signature for user %s", user.id)
        raise BillingError("that payment could not be verified")

    row.payment_id = payment_id
    row.status = "paid"

    # Extend from whichever is later, so paying early does not lose the days
    # already bought.
    now = dt.datetime.now(dt.timezone.utc)
    current = db._aware(user.tier_expires_at) if user.tier_expires_at else now
    start = max(current, now)
    user.tier = db.Tier.PREMIUM
    user.tier_expires_at = start + dt.timedelta(days=config.PREMIUM_DAYS)
    session.commit()
    logger.info("user %s is premium until %s", user.id, user.tier_expires_at)
    return user
