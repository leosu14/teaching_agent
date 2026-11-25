"""The live smoke run's review bundle: the generated lesson, kept encrypted for one human reviewer.

The live smoke run (deploy/live_smoke.py) tears its whole stack down at the end, and the repository is public, so a
workflow artifact can be downloaded by anyone signed in to GitHub. The lesson a teacher previewed and approved is
therefore kept as a *sealed* bundle: a zip of the teacher's preview, the learner's reading after approval, the
release decision, the lesson's images (as the teacher fetched them), the stage timings, the lesson's measured cost
and a blank review sheet, encrypted to the reviewer's RSA public key (`REVIEW_PUBLIC_KEY`, a variable of the
`staging` environment: a public key is not a secret). Only the holder of the matching private key can open it, on
their own machine:

    python deploy/review_bundle.py open live-smoke-review.enc.json --private-key ~/review.pem --out ./review

Encryption: a random 256-bit key encrypts the zip with AES-GCM; the key is wrapped with RSA-OAEP (SHA-256) for the
reviewer. The format string and the recipient's key fingerprint are authenticated (AES-GCM associated data), so a
bundle cannot be relabelled. The plaintext never touches the disk of the machine that seals it.

Make a key pair once, on the reviewer's machine (the private key never leaves it):

    openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:4096 -out ~/review.pem
    openssl pkey -in ~/review.pem -pubout          # paste this PEM as the REVIEW_PUBLIC_KEY variable
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import secrets
import sys
import zipfile
from pathlib import Path

FORMAT = "ta-live-smoke-review/1"
MIN_KEY_BITS = 3072
MAX_FILE_BYTES = 20 * 1024 * 1024  # what `open` accepts from one member (a lesson image is far smaller)
MAX_FILES = 200
MAX_TOTAL_BYTES = 100 * 1024 * 1024


class BundleError(Exception):
    """The key or the bundle is unusable; the message names what is wrong, never a key's value."""


# --- keys ----------------------------------------------------------------------------------------------------------


def load_public_key(pem: str):
    """The reviewer's RSA public key from PEM text (a GitHub variable may hold it with literal "\\n" escapes)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    text = (pem or "").strip().replace("\\n", "\n")
    if not text:
        raise BundleError("REVIEW_PUBLIC_KEY is not set (the reviewer's RSA public key, PEM)")
    try:
        key = serialization.load_pem_public_key(text.encode("ascii"))
    except (ValueError, UnicodeEncodeError, TypeError):
        raise BundleError("REVIEW_PUBLIC_KEY is not a PEM public key") from None
    if not isinstance(key, rsa.RSAPublicKey):
        raise BundleError("REVIEW_PUBLIC_KEY must be an RSA public key")
    if key.key_size < MIN_KEY_BITS:
        raise BundleError(f"REVIEW_PUBLIC_KEY must have at least {MIN_KEY_BITS} bits (got {key.key_size})")
    return key


def fingerprint(public_key) -> str:
    """SHA-256 of the key's DER SubjectPublicKeyInfo, hex: names the recipient without revealing anything."""
    from cryptography.hazmat.primitives import serialization

    der = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(der).hexdigest()


def _oaep():
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    return padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


def _aad(fp: str) -> bytes:
    return f"{FORMAT}|{fp}".encode()


# --- sealing and opening -------------------------------------------------------------------------------------------


def pack(files: dict[str, bytes]) -> bytes:
    """A zip of the bundle's files (names are relative paths; no directory parts that climb)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            _check_name(name)
            z.writestr(name, files[name])
    return buffer.getvalue()


def seal(files: dict[str, bytes], public_key) -> dict:
    """The encrypted envelope (a JSON-ready dict) of the files, for the holder of `public_key`'s private key."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    fp = fingerprint(public_key)
    data_key, nonce = AESGCM.generate_key(bit_length=256), secrets.token_bytes(12)
    ciphertext = AESGCM(data_key).encrypt(nonce, pack(files), _aad(fp))
    return {"format": FORMAT, "recipient_sha256": fp,
            "wrapped_key": base64.b64encode(public_key.encrypt(data_key, _oaep())).decode(),
            "nonce": base64.b64encode(nonce).decode(), "ciphertext": base64.b64encode(ciphertext).decode()}


def unseal(envelope: dict, private_key) -> dict[str, bytes]:
    """The files of an envelope, with the reviewer's private key. BundleError if it is not theirs or was altered."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not isinstance(envelope, dict) or envelope.get("format") != FORMAT:
        raise BundleError(f"not a {FORMAT} bundle")
    fp = fingerprint(private_key.public_key())
    if envelope.get("recipient_sha256") != fp:
        raise BundleError("this bundle was sealed for another key (recipient fingerprint differs)")
    try:
        data_key = private_key.decrypt(base64.b64decode(envelope["wrapped_key"]), _oaep())
        plain = AESGCM(data_key).decrypt(base64.b64decode(envelope["nonce"]),
                                         base64.b64decode(envelope["ciphertext"]), _aad(fp))
    except Exception:  # noqa: BLE001 - one uniform answer for a wrong key or an altered bundle
        raise BundleError("the bundle cannot be decrypted with this key (wrong key, or it was altered)") from None
    out: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(plain)) as z:
            infos = z.infolist()
            if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
                raise BundleError(f"the bundle holds more than {MAX_FILES} files or {MAX_TOTAL_BYTES} bytes")
            for info in infos:
                _check_name(info.filename)
                if info.file_size > MAX_FILE_BYTES:
                    raise BundleError(f"{info.filename} is larger than {MAX_FILE_BYTES} bytes")
                out[info.filename] = z.read(info)
    except zipfile.BadZipFile:
        raise BundleError("the bundle's content is not a readable zip") from None
    return out


def _check_name(name: str) -> None:
    parts = Path(name).parts
    if not name or name.startswith(("/", "\\")) or ".." in parts or any(":" in p for p in parts):
        raise BundleError(f"unsafe file name in the bundle: {name!r}")


# --- the review sheet ----------------------------------------------------------------------------------------------

RUBRIC = (
    ("Spanish accuracy", "no grammar, spelling or usage error a teacher would correct"),
    ("Level fit", "vocabulary and structures are A1; nothing beyond the objectives"),
    ("Objective alignment", "every objective is taught and practised"),
    ("Explanation clarity", "an adult beginner could follow it alone, in English as the instruction language"),
    ("Examples", "at least 3 natural, correct example exchanges"),
    ("Teaching question", "the graded question tests the objective; the grade matches what a teacher would give"),
    ("Safety and appropriateness", "nothing unsuitable; no stray web text"),
)


def render(preview: dict, release: dict, report: dict, images: list[str]) -> str:
    """REVIEW.md: the lesson as the teacher previewed it, what the run decided and measured, and a blank rubric."""
    lines = [f"# Live smoke lesson review: {preview.get('title') or '(untitled)'}", ""]
    lines += [f"- Level: {preview.get('level') or '-'}",
              f"- Lesson task: `{preview.get('task_id')}` · lesson artifact: `{preview.get('lesson_id')}`",
              (f"- Release: {release.get('status')} at {release.get('decided_at')} by the run's teacher identity "
               "(automated, through the teacher approval route). Your decision below is the human gate."),
              f"- Answer-bearing pieces withheld from this preview: {preview.get('withheld', 0)}", ""]
    if preview.get("objectives"):
        lines += ["## Objectives", "", *[f"- {o}" for o in preview["objectives"]], ""]
    if preview.get("introduction"):
        lines += ["## Introduction", "", *[f"{p}\n" for p in preview["introduction"]]]
    for section in preview.get("sections", []):
        lines += [f"## {section.get('heading') or section.get('purpose', 'Section')}", ""]
        lines += [f"{p}\n" for p in section.get("paragraphs", [])]
        if section.get("examples"):
            lines += ["Examples:", "", *[f"- {e}" for e in section["examples"]], ""]
        if section.get("analogy"):
            lines += [f"Analogy: {section['analogy']}", ""]
        for image in section.get("images", []):
            lines += [f"Image `{image.get('artifact_id')}`: {image.get('description') or ''}", ""]
    if preview.get("summary"):
        lines += ["## Summary", "", *[f"- {s}" for s in preview["summary"]], ""]
    if preview.get("references"):
        lines += ["## References", "", *[f"- {r.get('title')} ({r.get('publisher') or '-'}) {r.get('url') or ''}"
                                         for r in preview["references"]], ""]
    lines += ["## Images (as the teacher fetched them while the lesson was held)", ""]
    lines += [f"- `{name}`" for name in images] or ["- none"]
    lines += ["", "## What the run measured", "", "```json",
              json.dumps({k: report.get(k) for k in ("stages", "lesson_cost", "generation_server_seconds")},
                         indent=2, sort_keys=True), "```", ""]
    lines += ["## Human review (score 1 to 4; pass = no item below 3 and Spanish accuracy 4)", "",
              "Reviewer: ______  Date: ______", "", "| Item | 4 means | Score | Notes |", "|---|---|---|---|"]
    lines += [f"| {item} | {means} | | |" for item, means in RUBRIC]
    lines += ["", "Would you approve this lesson for a real learner as it is? yes / no", "",
              "Delete this folder once the review is recorded (it is the only plaintext copy).", ""]
    return "\n".join(lines)


# --- the reviewer's command ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open a live smoke review bundle with the reviewer's private key")
    sub = parser.add_subparsers(dest="action", required=True)
    opened = sub.add_parser("open", help="decrypt a bundle into a folder")
    opened.add_argument("bundle", type=Path)
    opened.add_argument("--private-key", type=Path, required=True, help="the reviewer's RSA private key (PEM)")
    opened.add_argument("--out", type=Path, required=True, help="a new or empty folder for the files")
    args = parser.parse_args(argv)
    from cryptography.hazmat.primitives import serialization

    try:
        envelope = json.loads(args.bundle.read_text(encoding="utf-8"))
        passphrase = os.environ.get("REVIEW_KEY_PASSPHRASE")
        private_key = serialization.load_pem_private_key(args.private_key.read_bytes(),
                                                         password=passphrase.encode() if passphrase else None)
        files = unseal(envelope, private_key)
        if args.out.exists() and any(args.out.iterdir()):
            raise BundleError(f"{args.out} is not empty")
        for name, body in files.items():
            target = args.out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
    except (BundleError, OSError, ValueError, TypeError) as exc:
        print(f"review bundle: {exc}", file=sys.stderr)
        return 1
    print(f"review bundle: {len(files)} files in {args.out} (start with REVIEW.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
