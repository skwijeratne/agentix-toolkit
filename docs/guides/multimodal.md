# Images, PDFs & audio

Agents aren't limited to text. You can send the model an **image** to look at, a
**PDF** to read, or an **audio** clip to listen to — as long as the model you're
using supports it. This is often called "multimodal" input (more than one *mode* of
content).

## Sending more than text

Normally a request is just a string. To include media, send a **list of parts**
instead — text mixed with images, documents, or audio:

```python
from agentix import TextPart, ImagePart

await agent.run([
    TextPart("What's in this picture?"),
    ImagePart.from_path("cat.png"),
])
```

The part types are `TextPart`, `ImagePart`, `DocumentPart` (for PDFs), and
`AudioPart`. You can build each from a local file, raw bytes, a URL, or
base64 data:

```python
ImagePart.from_path("diagram.png")          # a file (type detected automatically)
ImagePart.from_url("https://example.com/x.jpg")
DocumentPart.from_path("report.pdf")
```

## Each provider takes what it supports

Not every model accepts every kind of media. agentix translates each part into the
right format for your chosen provider — and if a provider *can't* handle something
(for example, Anthropic doesn't take audio), it raises a clear error instead of
silently dropping it, so you're never confused about what happened.

Plain text still works exactly as before — you only use parts when you have media
to send.

→ Runnable example:
[`examples/22_multimodal.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/22_multimodal.py)
