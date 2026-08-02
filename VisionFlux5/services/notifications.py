from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parseaddr
import re
import smtplib
from typing import Callable, Sequence

from pipeline.batch import format_elapsed

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@dataclass(frozen=True)
class EmailConfig:
    sender: str
    app_password: str
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    timeout_seconds: float = 20.0


@dataclass(frozen=True)
class CompletionReport:
    total_files: int
    succeeded_files: int
    failed_files: int
    elapsed_seconds: float
    completed_at: datetime | None = None
    failed_filenames: Sequence[str] = ()
    app_url: str | None = None


def validate_email_address(address: str) -> str:
    candidate = str(address or "").strip()
    _, parsed = parseaddr(candidate)
    if parsed != candidate or not _EMAIL_RE.fullmatch(candidate):
        raise ValueError("올바른 이메일 주소를 입력해 주세요.")
    return candidate


def build_completion_message(recipient: str, report: CompletionReport, *, sender: str = "") -> EmailMessage:
    recipient = validate_email_address(recipient)
    completed_at = report.completed_at or datetime.now(timezone.utc)
    failed = tuple(str(name) for name in report.failed_filenames if str(name))

    lines = [
        "VisionFlux 분석이 끝났습니다.",
        "",
        f"총 {int(report.total_files)}개 · 성공 {int(report.succeeded_files)}개 · 실패 {int(report.failed_files)}개",
        f"총 소요 시간: {format_elapsed(report.elapsed_seconds)}",
        f"완료 시각: {completed_at.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
    ]
    if failed:
        lines.extend(["", "실패한 파일:", *[f"- {name}" for name in failed]])
    if report.app_url:
        lines.extend(["", f"VisionFlux: {report.app_url}"])
    lines.extend(["", "이 메일은 VisionFlux 완료 알림으로 자동 발송되었습니다."])

    message = EmailMessage()
    message["Subject"] = "VisionFlux 분석 완료"
    message["To"] = recipient
    if sender:
        message["From"] = sender
    message.set_content("\n".join(lines))
    return message


def send_completion_email(
    config: EmailConfig,
    recipient: str,
    report: CompletionReport,
    *,
    smtp_factory: Callable[..., object] = smtplib.SMTP_SSL,
) -> None:
    sender = str(config.sender or "").strip()
    password = re.sub(r"\s+", "", str(config.app_password or ""))
    if not sender:
        raise ValueError("email sender 설정이 없습니다.")
    if not password:
        raise ValueError("email app_password 설정이 없습니다.")
    sender = validate_email_address(sender)
    message = build_completion_message(recipient, report, sender=sender)

    with smtp_factory(
        str(config.smtp_host),
        int(config.smtp_port),
        timeout=float(config.timeout_seconds),
    ) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)
