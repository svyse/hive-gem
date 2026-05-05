from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.memory.store import MemoryStore


log = logging.getLogger(__name__)


CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"


_CAPTION_LOCK = threading.Lock()
_CAPTION = {"processor": None, "model": None, "device": None}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    """Remove path separators and other dangerous characters."""

    name = (name or "upload").strip()
    name = name.replace("\\", "_").replace("/", "_").replace(":", "_")
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = name.strip("._ ")
    return name or "upload"


def _normalize_whitespace(text: str) -> str:
    t = (text or "").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 120) -> List[str]:
    """Simple char-based chunker with overlap."""

    text = _normalize_whitespace(text)
    if not text:
        return []

    chunk_size = max(200, int(chunk_size or 1200))
    overlap = max(0, min(int(overlap or 0), chunk_size // 2))

    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j].strip())
        if j >= n:
            break
        i = max(0, j - overlap)
    return [c for c in out if c]


def _extract_text_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pypdf is required to parse PDFs: {e}")

    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return _normalize_whitespace("\n\n".join(parts))


def _extract_text_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception as e:
        raise RuntimeError(f"python-docx is required to parse DOCX: {e}")

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text]
    return _normalize_whitespace("\n".join(parts))


def _load_caption_model() -> Tuple[Any, Any, str]:
    """Lazy-load BLIP captioning model."""

    # Cache across requests.
    with _CAPTION_LOCK:
        if _CAPTION["processor"] is not None and _CAPTION["model"] is not None:
            return _CAPTION["processor"], _CAPTION["model"], str(_CAPTION["device"])

        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except Exception as e:
            raise RuntimeError(f"Image captioning requires torch + transformers + pillow: {e}")

        # Keep captioning on CPU by default to avoid interfering with the main LLM.
        device = "cpu"
        processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID)
        model.to(device)
        model.eval()

        _CAPTION["processor"] = processor
        _CAPTION["model"] = model
        _CAPTION["device"] = device
        return processor, model, device


def _extract_text_image(path: Path) -> str:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pillow is required to parse images: {e}")

    processor, model, device = _load_caption_model()

    image = Image.open(str(path)).convert("RGB")

    # BLIP caption
    import torch

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    caption = processor.decode(out[0], skip_special_tokens=True)
    caption = _normalize_whitespace(caption)

    # We could add OCR here, but OCR tooling isn't always installed.
    return caption


def extract_text(path: Path, *, content_type: Optional[str] = None) -> str:
    """Best-effort extraction to plain text."""

    suffix = path.suffix.lower().strip(".")
    ctype = (content_type or "").lower()

    # Explicit types first
    if suffix in {"pdf"} or "application/pdf" in ctype:
        return _extract_text_pdf(path)
    if suffix in {"docx"}:
        return _extract_text_docx(path)
    if suffix in {"png", "jpg", "jpeg", "webp", "bmp"} or ctype.startswith("image/"):
        return _extract_text_image(path)

    # Plain text-ish
    if suffix in {"txt", "md", "csv", "log", "json", "yaml", "yml", "html", "htm"} or ctype.startswith("text/"):
        try:
            return _normalize_whitespace(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return _normalize_whitespace(path.read_text(errors="ignore"))

    # Fallback: try decode as text
    try:
        raw = path.read_bytes()
        return _normalize_whitespace(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return ""


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class UploadIngestResult:
    upload_id: str
    filename: str
    content_type: str
    size_bytes: int
    saved_path: str
    extracted_text_path: Optional[str]
    text_chars: int
    chunks_added: int
    training_examples_added: int
    memory_ids: List[int]


def _build_training_examples_from_chunks(
    *,
    filename: str,
    upload_id: str,
    chunks: List[str],
    max_examples: int,
) -> List[Tuple[str, str]]:
    """Generate SFT-style pairs from uploaded text.

    Strategy:
    - For documents: next-chunk prediction via an instruction-style prompt.
    - For single-chunk long texts: split into two halves and predict the second half.
    - For short single-chunk items (often images->caption): prompt->description.
    """

    out: List[Tuple[str, str]] = []
    if not chunks:
        return out

    max_examples = max(0, int(max_examples or 0))
    if max_examples == 0:
        return out

    if len(chunks) == 1:
        one = chunks[0]

        # If it's long enough, do a simple continuation split so the model can
        # learn the text without synthetic labels.
        if len(one) >= 500:
            split = max(200, len(one) // 2)
            prompt = f"""You are learning from an uploaded file for future Q&A.
File: {filename}
Upload ID: {upload_id}

Context (start of file):
{one[:split]}

Continue with the next part of the file:
"""
            completion = one[split:]
            return [(prompt, completion)][:max_examples]

        # For short items (often images -> caption), store a simple recall pair.
        prompt = f"""The user uploaded a file. Provide the best text description you can.
File: {filename}
Upload ID: {upload_id}

Description:
"""
        completion = one
        return [(prompt, completion)][:max_examples]

    # Next-chunk continuation
    for i in range(len(chunks) - 1):
        prompt = f"""You are learning from an uploaded file for future Q&A.
File: {filename}
Upload ID: {upload_id}

Context (excerpt):
{chunks[i]}

Continue with the next part of the file:
"""
        completion = chunks[i + 1]
        out.append((prompt, completion))
        if len(out) >= max_examples:
            break

    return out



def _ingest_sync(
    *,
    store: MemoryStore,
    data: bytes,
    filename: str,
    content_type: Optional[str],
    add_to_hive_memory: bool,
    create_training_examples: bool,
) -> UploadIngestResult:
    safe_name = _safe_filename(filename)

    # Create a stable-ish upload id (uuid + content hash prefix) to help deduplicate.
    up_id = uuid.uuid4().hex[:8] + "-" + _hash_bytes(data)

    upload_dir = Path(settings.uploads_dir) / up_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Try to preserve original extension when possible.
    ext = Path(safe_name).suffix
    if not ext and content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            ext = guessed

    saved_name = Path(safe_name).stem + (ext or "")
    saved_path = upload_dir / saved_name

    saved_path.write_bytes(data)

    # Extract
    extracted_text: str = ""
    extracted_text_path: Optional[str] = None
    try:
        extracted_text = extract_text(saved_path, content_type=content_type)
        extracted_text = _normalize_whitespace(extracted_text)
    except Exception as e:
        log.warning("Upload extraction failed for %s: %s", safe_name, e)
        extracted_text = ""

    if extracted_text:
        txt_path = upload_dir / "extracted.txt"
        try:
            txt_path.write_text(extracted_text, encoding="utf-8")
            extracted_text_path = str(txt_path)
        except Exception:
            extracted_text_path = None

    chunks = chunk_text(
        extracted_text,
        chunk_size=int(getattr(settings, "uploads_chunk_size_chars", 1200) or 1200),
        overlap=int(getattr(settings, "uploads_chunk_overlap_chars", 120) or 120),
    )

    if add_to_hive_memory and not chunks:
        chunks = [
            f"[UPLOAD:{up_id}] file={safe_name} type={content_type or 'unknown'}\n"
            f"Saved at: {saved_path}\n"
            "(Extraction produced no text.)",
        ]

    created_at = _now_iso()

    # Add to hive memory as chunked entries
    memory_ids: List[int] = []
    if add_to_hive_memory and chunks:
        for idx, chunk in enumerate(chunks, start=1):
            header = f"[UPLOAD:{up_id}] file={safe_name} type={content_type or 'unknown'} chunk={idx}/{len(chunks)}"
            content = header + "\n" + chunk
            mid = store.add(
                scope="hive",
                content=content,
                tags=["upload", f"upload:{up_id}", f"file:{safe_name}"],
                success=None,
                created_at=created_at,
            )
            memory_ids.append(mid)

    # Training examples
    train_n = 0
    if create_training_examples and bool(getattr(settings, "uploads_training_enabled", True)):
        max_ex = int(getattr(settings, "uploads_training_max_examples", 80) or 80)
        examples = _build_training_examples_from_chunks(
            filename=safe_name,
            upload_id=up_id,
            chunks=chunks,
            max_examples=max_ex,
        )
        for prompt, completion in examples:
            store.add_training_example(
                source="upload",
                upload_id=up_id,
                prompt=prompt,
                completion=completion,
                created_at=created_at,
            )
            train_n += 1

    # Upload record
    store.add_upload_record(
        upload_id=up_id,
        filename=safe_name,
        content_type=content_type or "",
        size_bytes=len(data),
        saved_path=str(saved_path),
        extracted_text_path=extracted_text_path,
        text_chars=len(extracted_text or ""),
        chunks_added=len(chunks),
        training_examples_added=train_n,
        created_at=created_at,
    )

    return UploadIngestResult(
        upload_id=up_id,
        filename=safe_name,
        content_type=content_type or "",
        size_bytes=len(data),
        saved_path=str(saved_path),
        extracted_text_path=extracted_text_path,
        text_chars=len(extracted_text or ""),
        chunks_added=len(chunks),
        training_examples_added=train_n,
        memory_ids=memory_ids,
    )


async def ingest_upload(
    *,
    store: MemoryStore,
    data: bytes,
    filename: str,
    content_type: Optional[str] = None,
    add_to_hive_memory: bool = True,
    create_training_examples: bool = True,
) -> UploadIngestResult:
    """Async wrapper to avoid blocking the FastAPI event loop."""

    return await asyncio.to_thread(
        _ingest_sync,
        store=store,
        data=data,
        filename=filename,
        content_type=content_type,
        add_to_hive_memory=add_to_hive_memory,
        create_training_examples=create_training_examples,
    )
