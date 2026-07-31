from datetime import datetime, timezone

import pytest

from services.notifications import (
    CompletionReport,
    EmailConfig,
    build_completion_message,
    send_completion_email,
    validate_email_address,
)


def test_validate_email_address_accepts_normal_address_and_rejects_bad_input():
    assert validate_email_address("skelethomas07@gmail.com") == "skelethomas07@gmail.com"
    with pytest.raises(ValueError):
        validate_email_address("not-an-email")


def test_build_completion_message_contains_counts_and_elapsed_time():
    report = CompletionReport(
        total_files=3,
        succeeded_files=2,
        failed_files=1,
        elapsed_seconds=125,
        completed_at=datetime(2026, 8, 1, 1, 30, tzinfo=timezone.utc),
        failed_filenames=("bad.tif",),
        app_url="https://example.streamlit.app",
    )
    message = build_completion_message("skelethomas07@gmail.com", report)
    body = message.get_body(preferencelist=("plain",)).get_content()
    assert message["Subject"] == "VisionFlux 분석 완료"
    assert "총 3개" in body
    assert "성공 2개" in body
    assert "실패 1개" in body
    assert "2분 05초" in body
    assert "bad.tif" in body
    assert "https://example.streamlit.app" in body


def test_send_completion_email_logs_in_and_sends_message():
    calls = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls.append(("connect", host, port, timeout))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def login(self, sender, password):
            calls.append(("login", sender, password))

        def send_message(self, message):
            calls.append(("send", message["To"], message["From"]))

    config = EmailConfig(sender="visionflux@gmail.com", app_password="abcd efgh")
    report = CompletionReport(total_files=1, succeeded_files=1, failed_files=0, elapsed_seconds=5)

    send_completion_email(
        config,
        "skelethomas07@gmail.com",
        report,
        smtp_factory=FakeSMTP,
    )

    assert calls[0][:3] == ("connect", "smtp.gmail.com", 465)
    assert calls[1] == ("login", "visionflux@gmail.com", "abcdefgh")
    assert calls[2] == ("send", "skelethomas07@gmail.com", "visionflux@gmail.com")


def test_send_completion_email_rejects_missing_credentials():
    config = EmailConfig(sender="", app_password="")
    report = CompletionReport(total_files=1, succeeded_files=1, failed_files=0, elapsed_seconds=1)
    with pytest.raises(ValueError, match="sender"):
        send_completion_email(config, "skelethomas07@gmail.com", report)
