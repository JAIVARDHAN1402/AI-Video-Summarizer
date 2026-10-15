"""Accounts, one-time codes, tiers and what they are allowed to spend.

The parts worth testing here are the ones where being wrong is not a bug but
a hole: a code that can be brute-forced, a sign-in that makes a second
account for the same person, an upgrade a client can award itself, and an
allowance that can be walked past.
"""

from __future__ import annotations

import datetime as dt

import pytest

from server import billing, config, db, identity, notify, quota


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A database of its own, so tests cannot see each other's accounts."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    db.Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    monkeypatch.setattr(db, "SessionLocal", maker)
    monkeypatch.setattr(db, "session", lambda: maker())
    monkeypatch.setattr(config, "DEV_OTP_ECHO", True)
    return maker()


@pytest.fixture
def sent(monkeypatch):
    """Capture the code instead of sending it."""
    box: dict[str, str] = {}
    monkeypatch.setattr(notify, "send_code",
                        lambda channel, destination, code: box.update(
                            {"channel": channel, "to": destination, "code": code}))
    return box


# --- one spelling, one account -------------------------------------------

def test_the_same_inbox_written_differently_is_one_account(session, sent, monkeypatch):
    # This is about identity, not about the resend window, so stand that down.
    monkeypatch.setattr(config, "OTP_RESEND_SEC", 0)
    identity.request_code(session, "email", "Foo@Example.com")
    a = identity.verify_code(session, "email", "Foo@Example.com", sent["code"])
    identity.request_code(session, "email", "  foo@example.COM ")
    b = identity.verify_code(session, "email", "foo@example.com", sent["code"])
    assert a.id == b.id
    assert session.query(db.User).count() == 1


def test_signing_in_with_google_finds_the_account_the_code_made(session, sent):
    """Otherwise the same person has two accounts and two allowances."""
    identity.request_code(session, "email", "student@vit.ac.in")
    by_code = identity.verify_code(session, "email", "student@vit.ac.in", sent["code"])
    by_google = identity._upsert(
        session, channel="email", destination="student@vit.ac.in",
        method="google", name="A Student")
    assert by_code.id == by_google.id
    assert session.query(db.User).count() == 1


def test_a_bare_indian_number_gets_its_country_code(session):
    assert identity.normalise("phone", "98765 43210") == "+919876543210"


# --- the code itself ------------------------------------------------------

def test_the_code_is_not_stored_in_the_clear(session, sent):
    identity.request_code(session, "email", "a@b.com")
    row = session.query(db.OtpCode).one()
    assert sent["code"] not in row.code_hash
    assert len(row.code_hash) == 64


def test_a_wrong_code_is_refused(session, sent):
    identity.request_code(session, "email", "a@b.com")
    wrong = "000000" if sent["code"] != "000000" else "111111"
    with pytest.raises(identity.IdentityError):
        identity.verify_code(session, "email", "a@b.com", wrong)


def test_guessing_is_cut_off_before_a_million_tries(session, sent, monkeypatch):
    """Six digits is a million guesses, which is nothing for a script."""
    monkeypatch.setattr(config, "OTP_MAX_ATTEMPTS", 3)
    identity.request_code(session, "email", "a@b.com")
    wrong = "000000" if sent["code"] != "000000" else "111111"
    for _ in range(3):
        with pytest.raises(identity.IdentityError):
            identity.verify_code(session, "email", "a@b.com", wrong)
    # Even the right code is no good now: the attempt budget is spent.
    with pytest.raises(identity.IdentityError, match="too many"):
        identity.verify_code(session, "email", "a@b.com", sent["code"])


def test_a_code_cannot_be_used_twice(session, sent):
    identity.request_code(session, "email", "a@b.com")
    identity.verify_code(session, "email", "a@b.com", sent["code"])
    with pytest.raises(identity.IdentityError):
        identity.verify_code(session, "email", "a@b.com", sent["code"])


def test_an_expired_code_is_refused(session, sent, monkeypatch):
    identity.request_code(session, "email", "a@b.com")
    row = session.query(db.OtpCode).one()
    row.expires_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
    session.commit()
    with pytest.raises(identity.IdentityError, match="expired"):
        identity.verify_code(session, "email", "a@b.com", sent["code"])


def test_the_endpoint_cannot_be_used_to_spam_somebody(session, sent):
    identity.request_code(session, "email", "victim@example.com")
    with pytest.raises(identity.IdentityError, match="just sent"):
        identity.request_code(session, "email", "victim@example.com")


def test_a_code_that_could_not_be_sent_cannot_be_used(session, monkeypatch):
    """Otherwise a failed send leaves a live code nobody received."""
    def refuse(channel, destination, code):
        raise notify.DeliveryError("no sender configured")

    monkeypatch.setattr(notify, "send_code", refuse)
    with pytest.raises(identity.IdentityError):
        identity.request_code(session, "email", "a@b.com")
    assert session.query(db.OtpCode).one().consumed is True


def test_an_unverified_google_address_is_refused():
    """Anyone can put someone else's address on a profile; only Google saying
    it verified it makes it theirs."""
    import inspect
    assert "email_verified" in inspect.getsource(identity.google_sign_in)


def test_static_key_deployment_does_not_advertise_a_broken_browser_sign_in(monkeypatch):
    """OTP delivery without a signing secret cannot create an API session."""
    from server.app import auth_methods

    monkeypatch.setattr(config, "JWT_SECRET", None)
    monkeypatch.setattr(config, "DEV_OTP_ECHO", True)
    monkeypatch.setattr(config, "SMTP_HOST", "")
    assert auth_methods()["email"] is False
    assert auth_methods()["phone"] is False

    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")
    assert auth_methods()["email"] is True
    assert auth_methods()["phone"] is True


def test_phone_sign_in_stays_hidden_until_a_complete_twilio_sender_exists(monkeypatch):
    from server.app import auth_methods

    monkeypatch.setattr(config, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(config, "DEV_OTP_ECHO", False)
    monkeypatch.setattr(config, "SMS_PROVIDER", "twilio")
    monkeypatch.setattr(config, "SMS_API_KEY", "token")
    monkeypatch.setattr(config, "SMS_TWILIO_SID", "AC123")
    monkeypatch.setattr(config, "SMS_TWILIO_FROM", "")
    assert auth_methods()["phone"] is False
    monkeypatch.setattr(config, "SMS_TWILIO_FROM", "+14155552671")
    assert auth_methods()["phone"] is True


# --- what a tier may do ---------------------------------------------------

def _user(session, tier=db.Tier.FREE):
    user = db.User(email=f"{tier.value}@x.com", tier=tier)
    session.add(user)
    session.commit()
    return user


def test_a_long_video_is_refused_before_anything_is_downloaded(session):
    user = _user(session)
    with pytest.raises(quota.QuotaError) as exc:
        quota.check(session, user, 11 * 3600)
    assert exc.value.reason == "video_too_long"
    assert exc.value.detail["limit_hours"] == config.FREE_MAX_VIDEO_HOURS


def test_premium_covers_what_free_does_not(session):
    user = _user(session, db.Tier.PREMIUM)
    assert quota.check(session, user, 11 * 3600) > 0


def test_cost_is_counted_in_calls_not_in_videos():
    """A 4-hour video is about five times a half-hour one, not eight."""
    short = quota.estimate_calls(0.5 * 3600)
    long = quota.estimate_calls(4 * 3600)
    assert short < long
    assert long < short * 8, "the window cap makes it sublinear, and so must the price"


def test_the_allowance_runs_out(session, monkeypatch):
    monkeypatch.setattr(config, "FREE_DAILY_CALLS", 10)
    monkeypatch.setitem(quota.TIERS, db.Tier.FREE,
                        quota.Allowance(config.FREE_MAX_VIDEO_HOURS * 3600, 10))
    user = _user(session)
    quota.charge(session, user, 9, kind="process")
    with pytest.raises(quota.QuotaError) as exc:
        quota.check(session, user, 3 * 3600)
    assert exc.value.reason == "daily_calls_exhausted"
    assert exc.value.detail["remaining"] == 1


def test_yesterday_does_not_count_against_today(session):
    user = _user(session)
    old = db.UsageEvent(user_id=user.id, calls=999,
                        created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2))
    session.add(old)
    session.commit()
    assert quota.used_today(session, user.id) == 0


def test_premium_that_has_run_out_is_free_again(session):
    user = _user(session, db.Tier.PREMIUM)
    user.tier_expires_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    session.commit()
    assert user.effective_tier is db.Tier.FREE
    with pytest.raises(quota.QuotaError):
        quota.check(session, user, 11 * 3600)


# --- paying ---------------------------------------------------------------

def test_a_client_cannot_award_itself_premium(session, monkeypatch):
    """The only evidence accepted is a signature the server recomputes."""
    monkeypatch.setattr(config, "RAZORPAY_KEY_ID", "rzp_test_x")
    monkeypatch.setattr(config, "RAZORPAY_KEY_SECRET", "secret")
    user = _user(session)
    session.add(db.Payment(user_id=user.id, order_id="order_1", amount_paise=19900))
    session.commit()

    with pytest.raises(billing.BillingError, match="could not be verified"):
        billing.confirm(session, user, "order_1", "pay_1", "not-a-real-signature")
    assert user.effective_tier is db.Tier.FREE


def test_a_real_signature_upgrades_the_account(session, monkeypatch):
    import hashlib
    import hmac

    monkeypatch.setattr(config, "RAZORPAY_KEY_ID", "rzp_test_x")
    monkeypatch.setattr(config, "RAZORPAY_KEY_SECRET", "secret")
    user = _user(session)
    session.add(db.Payment(user_id=user.id, order_id="order_1", amount_paise=19900))
    session.commit()

    signature = hmac.new(b"secret", b"order_1|pay_1", hashlib.sha256).hexdigest()
    billing.confirm(session, user, "order_1", "pay_1", signature)
    assert user.effective_tier is db.Tier.PREMIUM
    assert user.tier_expires_at is not None


def test_another_users_order_cannot_be_claimed(session, monkeypatch):
    monkeypatch.setattr(config, "RAZORPAY_KEY_ID", "rzp_test_x")
    monkeypatch.setattr(config, "RAZORPAY_KEY_SECRET", "secret")
    payer = _user(session, db.Tier.FREE)
    other = db.User(email="other@x.com")
    session.add_all([other, db.Payment(user_id=payer.id, order_id="order_1")])
    session.commit()

    with pytest.raises(billing.BillingError, match="does not belong"):
        billing.confirm(session, other, "order_1", "pay_1", "whatever")


def test_a_repeated_callback_does_not_extend_the_plan_twice(session, monkeypatch):
    import hashlib
    import hmac

    monkeypatch.setattr(config, "RAZORPAY_KEY_ID", "rzp_test_x")
    monkeypatch.setattr(config, "RAZORPAY_KEY_SECRET", "secret")
    user = _user(session)
    session.add(db.Payment(user_id=user.id, order_id="order_1"))
    session.commit()

    signature = hmac.new(b"secret", b"order_1|pay_1", hashlib.sha256).hexdigest()
    billing.confirm(session, user, "order_1", "pay_1", signature)
    first = user.tier_expires_at
    billing.confirm(session, user, "order_1", "pay_1", signature)
    assert user.tier_expires_at == first


# --- the resend window cannot be cleared by guessing ----------------------
# Found by driving the real sign-in form: the window was measured from the
# most recent *live* code, and six wrong guesses consume one. A script could
# alternate between guessing and resending and mail a stranger's inbox
# without limit, which is the one thing the window exists to stop.

def test_burning_the_attempts_does_not_unlock_another_send(session, sent, monkeypatch):
    monkeypatch.setattr(config, "OTP_MAX_ATTEMPTS", 3)
    identity.request_code(session, "email", "victim@example.com")

    for _ in range(config.OTP_MAX_ATTEMPTS + 1):
        with pytest.raises(identity.IdentityError):
            identity.verify_code(session, "email", "victim@example.com", "000000")

    assert session.query(db.OtpCode).one().consumed is True, "the code is spent"
    with pytest.raises(identity.IdentityError, match="just sent"):
        identity.request_code(session, "email", "victim@example.com")


def test_signing_in_does_not_unlock_another_send_either(session, sent):
    identity.request_code(session, "email", "someone@example.com")
    identity.verify_code(session, "email", "someone@example.com", sent["code"])
    with pytest.raises(identity.IdentityError, match="just sent"):
        identity.request_code(session, "email", "someone@example.com")


def test_a_new_code_retires_the_previous_one(session, sent, monkeypatch):
    """Two live codes for one inbox would double the guessing surface."""
    monkeypatch.setattr(config, "OTP_RESEND_SEC", 0)
    identity.request_code(session, "email", "a@b.com")
    first = sent["code"]
    identity.request_code(session, "email", "a@b.com")

    with pytest.raises(identity.IdentityError):
        identity.verify_code(session, "email", "a@b.com", first)
    # The newest one still works.
    assert identity.verify_code(session, "email", "a@b.com", sent["code"]) is not None


def test_the_window_opens_again_once_it_has_passed(session, sent, monkeypatch):
    monkeypatch.setattr(config, "OTP_RESEND_SEC", 0)
    identity.request_code(session, "email", "a@b.com")
    identity.request_code(session, "email", "a@b.com")   # no refusal
