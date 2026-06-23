"""Core multimodal content: parts, constructors, validation, and round-trip."""

from __future__ import annotations

import base64

import pytest

from agentix import (
    AudioPart,
    DocumentPart,
    ImagePart,
    Message,
    Role,
    TextPart,
    message_from_dict,
    message_to_dict,
)


def test_text_message_unchanged() -> None:
    msg = Message(Role.USER, "plain string")
    assert isinstance(msg.content, str)
    assert msg.text == "plain string"


def test_message_text_joins_text_parts_only() -> None:
    msg = Message(
        Role.USER,
        [TextPart("a "), ImagePart.from_base64("aGk=", "image/png"), TextPart("b")],
    )
    assert msg.text == "a b"  # media parts contribute no text


def test_from_bytes_and_to_bytes_round_trip() -> None:
    part = ImagePart.from_bytes(b"hello", "image/png", filename="h.png")
    assert part.media_type == "image/png"
    assert part.filename == "h.png"
    assert part.to_bytes() == b"hello"
    assert part.data_uri() == f"data:image/png;base64,{base64.b64encode(b'hello').decode()}"


def test_from_url_has_no_inline_data() -> None:
    part = ImagePart.from_url("https://example.com/cat.png")
    assert part.url == "https://example.com/cat.png"
    assert part.data is None
    with pytest.raises(ValueError):
        part.to_bytes()
    with pytest.raises(ValueError):
        part.data_uri()


def test_exactly_one_source_required() -> None:
    with pytest.raises(ValueError):
        ImagePart()  # neither data nor url
    with pytest.raises(ValueError):
        ImagePart(data="x", url="y", media_type="image/png")  # both


def test_inline_data_requires_media_type() -> None:
    with pytest.raises(ValueError):
        ImagePart(data="x")  # base64 data but no media_type


def test_from_path_infers_media_type(tmp_path: object) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "pic.png"
    p.write_bytes(b"\x89PNG\r\n")
    part = ImagePart.from_path(p)
    assert part.media_type == "image/png"
    assert part.filename == "pic.png"
    assert part.to_bytes() == b"\x89PNG\r\n"


def test_serde_round_trip_multimodal() -> None:
    msg = Message(
        Role.USER,
        [
            TextPart("look:"),
            ImagePart.from_base64("aGk=", "image/png"),
            DocumentPart.from_url("https://example.com/x.pdf", "application/pdf"),
            AudioPart.from_bytes(b"snd", "audio/wav"),
        ],
        trusted=True,
    )
    restored = message_from_dict(message_to_dict(msg))

    assert isinstance(restored.content, list)
    kinds = [type(p).__name__ for p in restored.content]
    assert kinds == ["TextPart", "ImagePart", "DocumentPart", "AudioPart"]
    assert restored.content[1].data == "aGk="  # type: ignore[union-attr]
    assert restored.content[2].url.endswith("x.pdf")  # type: ignore[union-attr]
    assert restored.content[3].to_bytes() == b"snd"  # type: ignore[union-attr]
    assert restored.text == "look:"


def test_serde_plain_string_still_plain() -> None:
    d = message_to_dict(Message(Role.USER, "hi"))
    assert d["content"] == "hi"
    assert message_from_dict(d).content == "hi"
