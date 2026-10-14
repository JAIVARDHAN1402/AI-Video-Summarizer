"""Delivering a one-time code to an inbox or a phone.

Each channel has a real sender and the same fallback: when nothing is
configured the code is written to the log instead of sent. That is what makes
the whole sign-in flow demonstrable on a laptop with no SMTP account and no
SMS credit — and it is also exactly the behaviour that must never reach a
deployment, so it is a deliberate setting rather than a quiet default, and
turning it off makes an unconfigured channel refuse instead of leak.
"""

from __future__ import annotations

import logging
import ssl

from server import config

logger = logging.getLogger("server.notify")


class DeliveryError(RuntimeError):
    """The code could not be delivered, so the caller must not pretend it was."""


def _echo(channel: str, destination: str, code: str) -> None:
    if not config.DEV_OTP_ECHO:
        raise DeliveryError(
            f"no {channel} sender is configured on this server, so a code "
            "cannot be delivered"
        )
    logger.warning(
        "DEV: no %s sender configured — the code for %s is %s "
        "(set VS_DEV_OTP_ECHO=0 to refuse instead)",
        channel, destination, code,
    )


def send_email_code(destination: str, code: str) -> None:
    if not (config.SMTP_HOST and config.SMTP_USER and config.SMTP_PASSWORD):
        _echo("email", destination, code)
        return

    import smtplib
    from email.message import EmailMessage

    message = EmailMessage()
    message["Subject"] = f"{code} is your sign-in code"
    message["From"] = config.SMTP_FROM or config.SMTP_USER
    message["To"] = destination
    message.set_content(
        f"Your sign-in code is {code}.\n\n"
        f"It is valid for {config.OTP_TTL_SEC // 60} minutes and can be used "
        "once. If you did not ask for it, you can ignore this message."
    )
    def _send_starttls() -> None:
        # Be explicit about the TLS handshake. Some providers (notably Gmail)
        # will close a session that attempts STARTTLS without an EHLO/context.
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
            smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.send_message(message)

    try:
        _send_starttls()
    except Exception as exc:
        # Gmail supports both STARTTLS on 587 and implicit TLS on 465. Corporate
        # networks occasionally interfere with the former, so transparently use
        # its official SSL endpoint as a one-time fallback.
        if config.SMTP_HOST.lower() != "smtp.gmail.com" or config.SMTP_PORT != 587:
            raise DeliveryError(f"could not send the email: {exc}") from exc
        try:
            with smtplib.SMTP_SSL(
                config.SMTP_HOST, 465, timeout=20,
                context=ssl.create_default_context(),
            ) as smtp:
                smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
                smtp.send_message(message)
        except Exception as ssl_exc:
            raise DeliveryError(
                "could not send the email through Gmail. Confirm that the Gmail "
                "account has 2-Step Verification enabled and that VS_SMTP_PASSWORD "
                "is a current 16-character App Password. "
                f"SMTP detail: {ssl_exc}"
            ) from ssl_exc
    logger.info("sent a sign-in code to %s", _masked(destination))


def send_sms_code(destination: str, code: str) -> None:
    if not config.SMS_PROVIDER or not config.SMS_API_KEY:
        _echo("sms", destination, code)
        return

    import requests

    text = f"{code} is your sign-in code. It expires in {config.OTP_TTL_SEC // 60} minutes."
    try:
        if config.SMS_PROVIDER == "msg91":
            if not config.SMS_SENDER:
                raise DeliveryError("MSG91 needs VS_SMS_SENDER")
            response = requests.post(
                "https://control.msg91.com/api/v5/flow/",
                headers={"authkey": config.SMS_API_KEY,
                         "Content-Type": "application/json"},
                json={"sender": config.SMS_SENDER,
                      "mobiles": destination.lstrip("+"),
                      "otp": code},
                timeout=20,
            )
        elif config.SMS_PROVIDER == "twilio":
            if not (config.SMS_TWILIO_SID and config.SMS_TWILIO_FROM):
                raise DeliveryError("Twilio needs VS_TWILIO_SID and VS_TWILIO_FROM")
            response = requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/"
                f"{config.SMS_TWILIO_SID}/Messages.json",
                auth=(config.SMS_TWILIO_SID, config.SMS_API_KEY),
                data={"To": destination, "From": config.SMS_TWILIO_FROM, "Body": text},
                timeout=20,
            )
        else:
            raise DeliveryError(f"unknown SMS provider {config.SMS_PROVIDER!r}")
    except DeliveryError:
        raise
    except Exception as exc:
        raise DeliveryError(f"could not send the SMS: {exc}") from exc

    if response.status_code >= 300:
        raise DeliveryError(
            f"the SMS provider refused the message "
            f"(HTTP {response.status_code}): {response.text[:200]}"
        )
    logger.info("sent a sign-in code to %s", _masked(destination))


def _masked(destination: str) -> str:
    """Enough of a destination to identify it in a log, not enough to use."""
    if "@" in destination:
        name, _, domain = destination.partition("@")
        return f"{name[:2]}***@{domain}"
    return f"***{destination[-3:]}"


def send_code(channel: str, destination: str, code: str) -> None:
    if channel == "email":
        send_email_code(destination, code)
    elif channel == "phone":
        send_sms_code(destination, code)
    else:
        raise DeliveryError(f"unknown channel {channel!r}")
