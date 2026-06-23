"""Multimodal content parts for :class:`~agentix.types.Message`.

A message's ``content`` is either a plain ``str`` (the common case, unchanged) or
a list of *parts* — text interleaved with media. Parts are framework-agnostic;
each provider adapter translates them into that vendor's block format and raises
a clear error for anything the provider can't accept.

Four part types::

    TextPart("describe this")
    ImagePart.from_path("cat.png")          # base64-encoded, mime inferred
    DocumentPart.from_url("https://…/x.pdf")
    AudioPart.from_bytes(raw, "audio/wav")

A media part holds **exactly one** of inline ``data`` (base64) or a remote
``url``. Build them with the ``from_*`` constructors rather than the raw fields.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar

T = TypeVar("T", bound="BinaryPart")


@dataclass
class TextPart:
    """A run of text within a multimodal message."""

    text: str
    kind: ClassVar[str] = "text"


@dataclass
class BinaryPart:
    """Base for media parts. Carries exactly one of ``data`` (base64) or ``url``.

    Prefer the constructors: :meth:`from_path`, :meth:`from_bytes`,
    :meth:`from_base64`, :meth:`from_url`.
    """

    data: str | None = None  # base64-encoded bytes
    url: str | None = None
    media_type: str = ""  # MIME type, e.g. "image/png" — required for inline data
    filename: str | None = None
    kind: ClassVar[str] = "binary"

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise ValueError("provide exactly one of data= (base64) or url=")
        if self.data is not None and not self.media_type:
            raise ValueError("media_type is required for inline (base64) data")

    @classmethod
    def from_base64(
        cls: type[T], data: str, media_type: str, *, filename: str | None = None
    ) -> T:
        return cls(data=data, media_type=media_type, filename=filename)

    @classmethod
    def from_bytes(
        cls: type[T], raw: bytes, media_type: str, *, filename: str | None = None
    ) -> T:
        encoded = base64.b64encode(raw).decode("ascii")
        return cls(data=encoded, media_type=media_type, filename=filename)

    @classmethod
    def from_url(
        cls: type[T], url: str, media_type: str = "", *, filename: str | None = None
    ) -> T:
        return cls(url=url, media_type=media_type, filename=filename)

    @classmethod
    def from_path(cls: type[T], path: str | os.PathLike[str]) -> T:
        """Read a local file, base64-encode it, and infer its MIME type."""
        p = Path(path)
        guessed, _ = mimetypes.guess_type(p.name)
        return cls(
            data=base64.b64encode(p.read_bytes()).decode("ascii"),
            media_type=guessed or "application/octet-stream",
            filename=p.name,
        )

    def to_bytes(self) -> bytes:
        """Decode inline data to raw bytes (raises for URL-only parts)."""
        if self.data is None:
            raise ValueError("part has no inline data (it is a URL reference)")
        return base64.b64decode(self.data)

    def data_uri(self) -> str:
        """``data:<media_type>;base64,<data>`` (raises for URL-only parts)."""
        if self.data is None:
            raise ValueError("data_uri() requires inline data, not a URL")
        return f"data:{self.media_type};base64,{self.data}"


@dataclass
class ImagePart(BinaryPart):
    """An image (PNG/JPEG/GIF/WebP) for vision-capable models."""

    kind: ClassVar[str] = "image"


@dataclass
class DocumentPart(BinaryPart):
    """A document (typically a PDF) for document-capable models."""

    kind: ClassVar[str] = "document"


@dataclass
class AudioPart(BinaryPart):
    """An audio clip for audio-capable models."""

    kind: ClassVar[str] = "audio"


#: Anything that can appear in a ``list`` message content.
ContentPart = TextPart | ImagePart | DocumentPart | AudioPart

_BINARY_KINDS: dict[str, type[ImagePart] | type[DocumentPart] | type[AudioPart]] = {
    "image": ImagePart,
    "document": DocumentPart,
    "audio": AudioPart,
}


def part_to_dict(part: ContentPart) -> dict[str, Any]:
    """Serialize a content part to a JSON-able dict (used by the codec)."""
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    return {
        "type": part.kind,
        "data": part.data,
        "url": part.url,
        "media_type": part.media_type,
        "filename": part.filename,
    }


def part_from_dict(d: dict[str, Any]) -> ContentPart:
    """Reconstruct a content part from :func:`part_to_dict` output."""
    kind = d.get("type")
    if kind == "text":
        return TextPart(text=d["text"])
    cls = _BINARY_KINDS.get(kind or "")
    if cls is None:
        raise ValueError(f"unknown content part type: {kind!r}")
    return cls(
        data=d.get("data"),
        url=d.get("url"),
        media_type=d.get("media_type", ""),
        filename=d.get("filename"),
    )
