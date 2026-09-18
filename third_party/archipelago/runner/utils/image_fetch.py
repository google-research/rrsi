# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fetch remote image URLs and re-encode them as data: URLs.

CLI-wrapping agent harnesses (codex_agent, claude_code_agent, gemini_cli_agent,
cursor_agent, …) all share the same problem: the CLI's image-input flag takes a
local file path, but task definitions in Studio commonly embed images as
`image_url` blocks with remote URLs (presigned S3, https). Rather than each
MCP server doing its own outbound HTTP — which would multiply SSRF surface,
egress policy, streaming/timeout handling, and per-protocol error handling —
this helper resolves remote URLs to base64-encoded data URLs on the harness
side. The MCP servers then only need to decode the data URL and write bytes
to disk.

The 5MB-per-image cap matches Codex CLI's documented soft guideline and is
a reasonable upper bound for the model context budget of any vision model.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
from loguru import logger

MAX_IMAGE_BYTES = 5 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 30.0
# httpx.AsyncClient's default is 20; we want enough hops to traverse
# legitimate CDN/presigned-URL chains but not unbounded.
MAX_REDIRECTS = 10

# Allowed schemes for inline pass-through (no fetch needed).
_DATA_SCHEMES = ("data:",)
# Schemes the helper will resolve to bytes via outbound HTTP.
_HTTPS_SCHEMES = ("https://",)


class ImageFetchError(Exception):
    """Raised when an image URL cannot be resolved within budget."""


def _sniff_image_mime(data: bytes) -> str | None:
    """Identify a known image format from the leading bytes.

    Returns the MIME type ("image/png", "image/jpeg", "image/gif",
    "image/webp") if the magic bytes match a supported format, else None.

    Used as a fallback when the upstream response advertises a generic
    content-type (e.g. S3 stores objects as `binary/octet-stream` when no
    ContentType was set on upload — see
    rl-studio/server/packages/custom_field_files/service.py upload_files()).
    Without this, every downstream MCP server's `_extension_for_mime` maps
    the wrong MIME to `.bin`, and the resulting file is rejected by Gemini
    CLI's `@path` resolver (extension-based via mime/lite) and by Studio's
    `mcp__gateway__filesystem_read_image_file` (extension allowlist).
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    # WEBP = RIFF<size>WEBP; need 12 bytes to confirm.
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _fetch_streaming(
    client: httpx.AsyncClient, url: str
) -> tuple[bytes, str | None]:
    """GET `url`, following redirects manually with per-hop https validation.

    `client.stream("GET", ...)` is invoked with redirects disabled at the
    client level (see resolve_to_data_urls). We follow them ourselves so we
    can reject any hop that tries to land on a non-https URL — otherwise an
    https→http://internal-host redirect would silently bypass the SSRF
    mitigation and the docstring's https-only guarantee.

    Body is streamed with an incremental byte counter; the connection is
    aborted past MAX_IMAGE_BYTES so a hostile or misconfigured source can't
    blow memory.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not current.startswith(_HTTPS_SCHEMES):
            raise ImageFetchError(
                f"Refusing to follow redirect to non-https URL: {current[:80]}"
            )
        async with client.stream(
            "GET", current, timeout=FETCH_TIMEOUT_SECONDS
        ) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ImageFetchError(
                        f"Redirect from {current[:80]} has no Location header"
                    )
                # Resolve relative locations against the current URL.
                current = str(httpx.URL(current).join(location))
                continue

            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ImageFetchError(
                        f"Image exceeds {MAX_IMAGE_BYTES}-byte cap during fetch from {url[:80]}"
                    )
                chunks.append(chunk)
            raw = b"".join(chunks)
            mime = (
                response.headers.get("content-type", "").split(";")[0].strip() or None
            )
            # If the response advertised a generic MIME (e.g. S3's
            # binary/octet-stream default when no ContentType was set on
            # upload), sniff magic bytes for a real image format. Only
            # override when sniff is confident — leave non-image bytes
            # alone so we don't mislabel something.
            if not mime or not mime.startswith("image/"):
                sniffed = _sniff_image_mime(raw[:16])
                if sniffed is not None:
                    mime = sniffed
            return raw, mime

    raise ImageFetchError(
        f"Too many redirects (>{MAX_REDIRECTS}) starting from {url[:80]}"
    )


def _to_data_url(raw: bytes, mime: str | None) -> str:
    mime = mime or "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def resolve_to_data_urls(
    images: list[dict[str, Any]],
    *,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Rewrite every entry in `images` so its `url` is a `data:` URL.

    - `data:...` entries pass through unchanged.
    - `https://...` entries are fetched (streamed, size-capped) and re-encoded.
    - Other schemes (http, s3, file, …) are rejected with ImageFetchError so the
      caller sees a clear contract violation rather than a silent CLI failure
      downstream.

    Each entry's other keys (e.g. `detail`) are preserved.
    """
    if not images:
        return []

    owns_client = client is None
    if client is None:
        # follow_redirects=False so _fetch_streaming can validate the scheme
        # at each hop and refuse https → http redirects (SSRF guard).
        client = httpx.AsyncClient(follow_redirects=False)

    resolved: list[dict[str, Any]] = []
    try:
        for i, image in enumerate(images):
            url = image.get("url")
            if not isinstance(url, str) or not url:
                raise ImageFetchError(f"Image {i} has no url")

            if url.startswith(_DATA_SCHEMES):
                resolved.append(dict(image))
                continue

            if url.startswith(_HTTPS_SCHEMES):
                try:
                    raw, mime = await _fetch_streaming(client, url)
                except (httpx.HTTPError, httpx.InvalidURL) as exc:
                    raise ImageFetchError(
                        f"Failed to fetch image {i} from {url[:80]}: {exc}"
                    ) from exc
                logger.info(
                    f"image_fetch: resolved {url[:80]} -> data url ({len(raw)} bytes, mime={mime})"
                )
                new_entry = dict(image)
                new_entry["url"] = _to_data_url(raw, mime)
                resolved.append(new_entry)
                continue

            raise ImageFetchError(
                f"Unsupported image URL scheme for image {i}: "
                f"{url[:40]}... (expected data: or https:)"
            )
    finally:
        if owns_client:
            await client.aclose()

    return resolved
