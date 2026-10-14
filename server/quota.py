"""What a video costs a user, and whether they can afford it.

The allowance is counted in model calls, not in videos. A four-hour video is
about twenty-two calls and a half-hour one about four, so a limit of "three
videos a day" would overcharge every short video and undercharge exactly the
long ones the limit exists for. Calls are also the thing that actually runs
out: the providers meter per day per account, and a day's allowance is spent
by roughly one long video.

The estimate is not a guess. It calls the same windowing the pipeline will
call, so if the window settings change the price changes with them.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional

import config as core_config
from core import understand
from server import config, db

logger = logging.getLogger("server.quota")


@dataclass(frozen=True)
class Allowance:
    """What one tier may do."""

    max_video_sec: float
    daily_calls: int

    @property
    def max_video_hours(self) -> float:
        return self.max_video_sec / 3600.0


TIERS: dict[db.Tier, Allowance] = {
    db.Tier.FREE: Allowance(
        max_video_sec=config.FREE_MAX_VIDEO_HOURS * 3600,
        daily_calls=config.FREE_DAILY_CALLS,
    ),
    db.Tier.PREMIUM: Allowance(
        max_video_sec=config.PREMIUM_MAX_VIDEO_HOURS * 3600,
        daily_calls=config.PREMIUM_DAILY_CALLS,
    ),
}


def allowance_for(user: Optional[db.User]) -> Allowance:
    if user is None:
        return TIERS[db.Tier.FREE]
    return TIERS[user.effective_tier]


def estimate_calls(duration_sec: float, *, with_extras: bool = True) -> int:
    """Model calls one video will cost, from the same code that will run it.

    ``with_extras`` counts the artefacts a reader usually goes on to ask for —
    the problem list, a quiz, an audio overview. They are generated on demand
    rather than during the run, but they come out of the same daily allowance,
    and a limit that ignored them would be spent without the reader being told
    why.
    """
    duration_sec = max(float(duration_sec or 0.0), 0.0)
    minutes = duration_sec / 60.0

    if minutes <= core_config.UNDERSTANDING_SINGLE_PASS_MAX_MIN:
        calls = 1
    else:
        windows = understand._windows(
            duration_sec,
            core_config.UNDERSTANDING_WINDOW_MIN,
            core_config.UNDERSTANDING_MAX_WINDOWS,
        )
        calls = len(windows) + 1          # the windows, plus the reduce pass

    if with_extras:
        problems = min(core_config.PROBLEMS_MAX_WINDOWS, max(1, round(calls * 0.9)))
        calls += problems + 2             # quiz, audio overview

    return int(calls)


def used_today(session, user_id: str, *, now: Optional[dt.datetime] = None) -> int:
    """Calls charged to this user since midnight UTC.

    Midnight UTC rather than local midnight because that is when the provider
    allowances reset; a limit on a different clock to the thing it is
    rationing would let a user spend tomorrow's quota tonight.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    since = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = (
        session.query(db.UsageEvent)
        .filter(db.UsageEvent.user_id == user_id, db.UsageEvent.created_at >= since)
        .all()
    )
    return sum(int(r.calls or 0) for r in rows)


def resets_at(now: Optional[dt.datetime] = None) -> dt.datetime:
    now = now or dt.datetime.now(dt.timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + dt.timedelta(days=1)


class QuotaError(Exception):
    """A request that the user's tier does not allow.

    Carries the numbers rather than only a sentence, so the client can show
    what the limit is and what it would take to get past it instead of a bare
    refusal.
    """

    def __init__(self, reason: str, message: str, **detail):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        return {"reason": self.reason, "message": self.message, **self.detail}


def check(session, user: Optional[db.User], duration_sec: float) -> int:
    """Whether this user may process this video, and what it will cost them.

    Returns the estimated calls. Raises :class:`QuotaError` otherwise.
    """
    allowance = allowance_for(user)
    cost = estimate_calls(duration_sec)

    if duration_sec > allowance.max_video_sec:
        raise QuotaError(
            "video_too_long",
            f"This video is {duration_sec / 3600:.1f} hours. Your plan covers "
            f"videos up to {allowance.max_video_hours:.0f} hours.",
            duration_hours=round(duration_sec / 3600, 2),
            limit_hours=allowance.max_video_hours,
            tier=(user.effective_tier.value if user else "free"),
        )

    if user is None:
        return cost

    used = used_today(session, user.id)
    if used + cost > allowance.daily_calls:
        raise QuotaError(
            "daily_calls_exhausted",
            f"This video needs about {cost} model calls and you have "
            f"{max(allowance.daily_calls - used, 0)} left today.",
            needed=cost,
            used=used,
            daily_calls=allowance.daily_calls,
            remaining=max(allowance.daily_calls - used, 0),
            resets_at=resets_at().isoformat(),
            tier=user.effective_tier.value,
        )
    return cost


def charge(session, user: Optional[db.User], calls: int, *,
           job_id: Optional[str] = None, kind: str = "process",
           duration_sec: float = 0.0) -> None:
    """Record what a run cost. A no-op when nobody is signed in."""
    if user is None or calls <= 0:
        return
    session.add(db.UsageEvent(
        user_id=user.id, job_id=job_id, kind=kind,
        calls=int(calls), duration_sec=float(duration_sec or 0.0),
    ))
    session.commit()


def summary(session, user: Optional[db.User]) -> dict:
    """What the client needs to show the reader where they stand."""
    allowance = allowance_for(user)
    used = used_today(session, user.id) if user is not None else 0
    return {
        "tier": (user.effective_tier.value if user else "free"),
        "max_video_hours": allowance.max_video_hours,
        "daily_calls": allowance.daily_calls,
        "used_today": used,
        "remaining": max(allowance.daily_calls - used, 0),
        "resets_at": resets_at().isoformat(),
    }
