import email
import imaplib
from email.header import decode_header
from email.utils import parseaddr
from typing import Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel


class EmailResponse(BaseModel):
    id: str
    subject: str
    body: str
    sender_email: str
    sender_display_name: Optional[str]
    sent_datetime: str


def decode_text(value: str) -> str:
    if not value:
        return ""

    parts = decode_header(value)
    text = ""

    for part, encoding in parts:
        if isinstance(part, bytes):
            text += part.decode(
                encoding or "utf-8",
                errors="ignore"
            )
        else:
            text += part
    return text


def extract_body(msg) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if "attachment" in str(part.get("Content-Disposition")):
                continue

            content_type = part.get_content_type()
            payload = part.get_payload(decode=True)

            if payload is None:
                continue

            text = payload.decode(
                part.get_content_charset() or "utf-8",
                errors="ignore"
            )

            if content_type == "text/plain":
                return text

            # use bs4 to parse html into text
            if content_type == "text/html":
                body = BeautifulSoup(text, "html.parser").get_text()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(
                msg.get_content_charset() or "utf-8",
                errors="ignore"
            )
    return body


def get_emails(host: str, username: str, password: str, limit: int = 25) -> list[EmailResponse]:
    connection = imaplib.IMAP4_SSL(host)
    connection.login(username, password)
    connection.select("INBOX")

    _, data = connection.search(None, "ALL")
    email_ids = data[0].split()
    emails = []

    for email_id in reversed(email_ids[-limit:]):
        _, message = connection.fetch(email_id, "(RFC822)")
        msg = email.message_from_bytes(message[0][1])

        display_name, sender = parseaddr(msg.get("From", ""))
        emails.append(
            EmailResponse(
                id=email_id.decode(),
                subject=decode_text(msg.get("Subject", "")),
                body=extract_body(msg=msg),
                sender_email=sender,
                sender_display_name=display_name,
                sent_datetime=msg.get("Date", ""),
            )
        )

    connection.logout()
    return emails