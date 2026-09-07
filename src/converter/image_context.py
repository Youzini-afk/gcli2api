"""Render Gemini request text parts as PNG image parts.

This module only handles the project's internal request envelope::

    {"model": "...", "request": {"contents": [...], ...}}

The private mode flag is consumed here so it can never be forwarded to the
upstream API.  Conversion is transactional: if any image cannot be rendered,
the caller gets an untouched, flag-free request that is safe to retry.
"""

from __future__ import annotations

import base64
import copy
import io
import os
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from src.converter.tool_anti_truncation import NON_TEXT_MODEL_PATTERN

IMAGE_CONTEXT_MODE_FLAG = "_image_context_mode"
IMAGE_CONTEXT_FONT_ENV = "GCLI2API_IMAGE_CONTEXT_FONT"


class ImageContextMode(str, Enum):
    """Supported scopes for turning request text into image context."""

    ALL_TEXT = "all_text"
    CURRENT_TURN = "current_turn"


@dataclass(frozen=True, slots=True)
class ImageContextResult:
    """Prepared request plus information needed for transparent fallback.

    ``original_payload`` is produced only when the private flag was present.
    It is a flag-free deep copy from before image conversion.  Callers may
    therefore reuse ``payload`` for ordinary retries and switch to
    ``original_payload`` if the upstream rejects multimodal input.
    """

    payload: dict[str, Any]
    original_payload: dict[str, Any] | None
    mode: ImageContextMode | None = None
    converted_parts: int = 0
    fallback_stage: str | None = None
    fallback_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.mode is not None

    @property
    def applied(self) -> bool:
        return self.converted_parts > 0 and self.fallback_reason is None

    @property
    def used_fallback(self) -> bool:
        return self.fallback_reason is not None


@dataclass(frozen=True, slots=True)
class _RenderStyle:
    width: int = 1600
    horizontal_padding: int = 56
    vertical_padding: int = 48
    body_font_size: int = 28
    header_font_size: int = 24
    line_gap: int = 10
    header_gap: int = 24
    tab_size: int = 4
    max_lines_per_page: int = 48


_STYLE = _RenderStyle()

# Fields that identify a part as a non-text Gemini Part.  A malformed part
# containing both one of these and ``text`` is deliberately left untouched so
# a function call, native image, or other structured input is never destroyed.
_NON_TEXT_DATA_FIELDS = frozenset(
    {
        "inlineData",
        "inline_data",
        "fileData",
        "file_data",
        "functionCall",
        "function_call",
        "functionResponse",
        "function_response",
        "executableCode",
        "executable_code",
        "codeExecutionResult",
        "code_execution_result",
    }
)


def _font_candidates() -> Iterable[str]:
    configured = os.getenv(IMAGE_CONTEXT_FONT_ENV)
    if configured:
        yield configured

    # Prefer fonts with broad CJK coverage.  Noto Sans Mono CJK also keeps
    # source-code columns predictable when it is installed.
    yield from (
        "/usr/share/fonts/opentype/noto/NotoSansMonoCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        # Pillow commonly ships or can resolve DejaVu.  It is a final portable
        # fallback, although its CJK coverage is less complete than Noto.
        "DejaVuSansMono.ttf",
        "DejaVuSans.ttf",
    )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _font_candidates():
        if Path(candidate).is_absolute() and not Path(candidate).is_file():
            continue
        try:
            return ImageFont.truetype(candidate, size=size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def _is_cluster_extension(character: str) -> bool:
    codepoint = ord(character)
    return (
        bool(unicodedata.combining(character))
        or unicodedata.category(character) in {"Mn", "Mc", "Me"}
        or 0xFE00 <= codepoint <= 0xFE0F  # variation selectors
        or 0xE0100 <= codepoint <= 0xE01EF
        or 0x1F3FB <= codepoint <= 0x1F3FF  # emoji skin-tone modifiers
    )


def _text_clusters(text: str) -> list[str]:
    """Group common combining and ZWJ sequences before line wrapping."""

    clusters: list[str] = []
    for character in text:
        if not clusters:
            clusters.append(character)
            continue
        if (
            _is_cluster_extension(character)
            or character == "\u200d"
            or clusters[-1].endswith("\u200d")
        ):
            clusters[-1] += character
        else:
            clusters.append(character)
    return clusters


def _pixel_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if not text:
        return 0.0
    return float(draw.textlength(text, font=font))


def _wrap_line_by_pixel_width(
    line: str,
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
    tab_size: int = _STYLE.tab_size,
) -> list[str]:
    """Wrap one logical line without discarding any visible whitespace.

    Tabs are expanded to fixed tab stops before measurement.  Each break is
    chosen using Pillow's actual font metrics rather than a character-count
    estimate.  An individual glyph wider than the canvas remains on its own
    line so no content is silently removed.
    """

    expanded = line.expandtabs(tab_size)
    if not expanded:
        return [""]

    clusters = _text_clusters(expanded)
    wrapped: list[str] = []
    start = 0
    while start < len(clusters):
        # Find a small upper bound exponentially before binary searching.  A
        # binary search over the entire remaining string on every visual line
        # turns very long one-line prompts into quadratic work.
        best = start
        step = 1
        high = min(len(clusters), start + step)
        while True:
            candidate = "".join(clusters[start:high])
            if _pixel_width(draw, candidate, font) > max_width:
                break
            best = high
            if high == len(clusters):
                break
            step *= 2
            high = min(len(clusters), start + step)

        if best == len(clusters):
            wrapped.append("".join(clusters[start:best]))
            break

        low = best + 1
        binary_high = high - 1
        while low <= binary_high:
            middle = (low + binary_high) // 2
            candidate = "".join(clusters[start:middle])
            if _pixel_width(draw, candidate, font) <= max_width:
                best = middle
                low = middle + 1
            else:
                binary_high = middle - 1

        if best == start:
            best = start + 1
        wrapped.append("".join(clusters[start:best]))
        start = best
    return wrapped


def _line_height(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    line_gap: int,
) -> int:
    bounding_box = draw.textbbox((0, 0), "Ag国", font=font)
    return max(1, bounding_box[3] - bounding_box[1]) + line_gap


def _layout_text_lines(
    text: str,
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
    tab_size: int = _STYLE.tab_size,
) -> list[str]:
    """Return every hard and soft-wrapped visual line in source order."""

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    rendered_lines: list[str] = []
    for logical_line in normalized_text.split("\n"):
        rendered_lines.extend(
            _wrap_line_by_pixel_width(
                logical_line,
                draw=draw,
                font=font,
                max_width=max_width,
                tab_size=tab_size,
            )
        )
    return rendered_lines


def _paginate_lines(lines: list[str], max_lines_per_page: int) -> list[list[str]]:
    """Split visual lines into pages without imposing a total page limit."""

    if max_lines_per_page < 1:
        raise ValueError("max_lines_per_page must be positive")
    return [
        lines[start : start + max_lines_per_page]
        for start in range(0, len(lines), max_lines_per_page)
    ]


def _page_header(role: str, part_index: int, page_index: int, page_count: int) -> str:
    return (
        f"role={role or 'unknown'} | part={part_index + 1} " f"| page={page_index + 1}/{page_count}"
    )


def _render_text_png_pages(text: str, *, role: str, part_index: int) -> list[bytes]:
    """Render a text part to one or more readable PNG pages."""

    style = _STYLE
    body_font = _load_font(style.body_font_size)
    header_font = _load_font(style.header_font_size)
    measuring_image = Image.new("RGB", (style.width, 1), "white")
    measuring_draw = ImageDraw.Draw(measuring_image)
    content_width = style.width - (2 * style.horizontal_padding)
    rendered_lines = _layout_text_lines(
        text,
        draw=measuring_draw,
        font=body_font,
        max_width=content_width,
        tab_size=style.tab_size,
    )
    pages = _paginate_lines(rendered_lines, style.max_lines_per_page)
    body_line_height = _line_height(measuring_draw, body_font, style.line_gap)
    header_line_height = _line_height(measuring_draw, header_font, style.line_gap)
    png_pages: list[bytes] = []
    page_count = len(pages)
    for page_index, page_lines in enumerate(pages):
        height = (
            (2 * style.vertical_padding)
            + header_line_height
            + style.header_gap
            + (body_line_height * len(page_lines))
        )
        image = Image.new("RGB", (style.width, height), "white")
        draw = ImageDraw.Draw(image)
        header_bottom = style.vertical_padding + header_line_height + (style.header_gap // 2)
        draw.rectangle((0, 0, style.width, header_bottom), fill=(241, 245, 249))
        draw.text(
            (style.horizontal_padding, style.vertical_padding),
            _page_header(role, part_index, page_index, page_count),
            fill=(51, 65, 85),
            font=header_font,
        )
        draw.line(
            (
                style.horizontal_padding,
                header_bottom,
                style.width - style.horizontal_padding,
                header_bottom,
            ),
            fill=(203, 213, 225),
            width=2,
        )

        y_position = style.vertical_padding + header_line_height + style.header_gap
        for line in page_lines:
            draw.text(
                (style.horizontal_padding, y_position),
                line,
                fill=(15, 23, 42),
                font=body_font,
            )
            y_position += body_line_height

        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        png_pages.append(output.getvalue())
    return png_pages


def _is_convertible_text_part(part: Any) -> bool:
    if not isinstance(part, dict) or _NON_TEXT_DATA_FIELDS.intersection(part):
        return False
    text = part.get("text")
    return isinstance(text, str) and bool(text.strip())


def _convert_parts(parts: Any, *, role: str) -> int:
    if not isinstance(parts, list):
        return 0

    converted = 0
    expanded_parts: list[Any] = []
    for part_index, part in enumerate(parts):
        if not _is_convertible_text_part(part):
            expanded_parts.append(part)
            continue
        png_pages = _render_text_png_pages(part["text"], role=role, part_index=part_index)
        retained_fields = {key: value for key, value in part.items() if key != "text"}
        for png_bytes in png_pages:
            image_part = retained_fields.copy()
            image_part["inlineData"] = {
                "mimeType": "image/png",
                "data": base64.b64encode(png_bytes).decode("ascii"),
            }
            expanded_parts.append(image_part)
        converted += 1
    parts[:] = expanded_parts
    return converted


def _convert_request(request: dict[str, Any], mode: ImageContextMode) -> int:
    converted = 0
    if mode is ImageContextMode.ALL_TEXT:
        system_instruction = request.get("systemInstruction")
        if isinstance(system_instruction, dict):
            converted += _convert_parts(
                system_instruction.get("parts"),
                role=str(system_instruction.get("role") or "system"),
            )

    contents = request.get("contents")
    if not isinstance(contents, list):
        return converted

    first_content = 0
    if mode is ImageContextMode.CURRENT_TURN:
        last_model_content = -1
        for content_index, content in enumerate(contents):
            if isinstance(content, dict) and content.get("role") == "model":
                last_model_content = content_index
        first_content = last_model_content + 1

    for content in contents[first_content:]:
        if not isinstance(content, dict):
            continue
        converted += _convert_parts(
            content.get("parts"),
            role=str(content.get("role") or "unknown"),
        )
    return converted


def _coerce_mode(value: Any) -> ImageContextMode | None:
    if isinstance(value, ImageContextMode):
        return value
    if not isinstance(value, str):
        return None
    try:
        return ImageContextMode(value)
    except ValueError:
        return None


def _exception_reason(prefix: str, error: Exception) -> str:
    detail = str(error).strip()
    suffix = f": {detail}" if detail else ""
    return f"{prefix}: {type(error).__name__}{suffix}"


def prepare_image_context_request(body: dict[str, Any]) -> ImageContextResult:
    """Consume the private mode and prepare a reusable Gemini request payload.

    A body without ``IMAGE_CONTEXT_MODE_FLAG`` is the fast path and is returned
    by identity without copying.  Once the flag is present, it is always removed
    before a payload is returned, including invalid modes and every fallback.
    """

    if IMAGE_CONTEXT_MODE_FLAG not in body:
        return ImageContextResult(payload=body, original_payload=None)

    raw_mode = body.get(IMAGE_CONTEXT_MODE_FLAG)
    mode = _coerce_mode(raw_mode)
    try:
        original_payload = copy.deepcopy(body)
    except Exception as error:
        # JSON request bodies are normally deepcopy-safe.  Retain the invariant
        # that the private control field is consumed even for unusual mappings.
        original_payload = dict(body)
        original_payload.pop(IMAGE_CONTEXT_MODE_FLAG, None)
        return ImageContextResult(
            payload=original_payload,
            original_payload=original_payload,
            mode=mode,
            fallback_stage="copy",
            fallback_reason=_exception_reason("could not copy image-context request", error),
        )

    original_payload.pop(IMAGE_CONTEXT_MODE_FLAG, None)
    if mode is None:
        return ImageContextResult(
            payload=original_payload,
            original_payload=original_payload,
            fallback_stage="mode_validation",
            fallback_reason=f"unsupported image context mode: {raw_mode!r}",
        )

    model_name = str(original_payload.get("model", ""))
    if NON_TEXT_MODEL_PATTERN.search(model_name):
        return ImageContextResult(
            payload=original_payload,
            original_payload=original_payload,
            mode=mode,
            fallback_stage="model_filter",
            fallback_reason=f"image context is not applied to non-text model: {model_name}",
        )

    request = original_payload.get("request")
    if not isinstance(request, dict):
        return ImageContextResult(
            payload=original_payload,
            original_payload=original_payload,
            mode=mode,
            fallback_stage="request_validation",
            fallback_reason="image context request must contain a request object",
        )

    try:
        transformed_payload = copy.deepcopy(original_payload)
        transformed_request = transformed_payload["request"]
        converted_parts = _convert_request(transformed_request, mode)
    except Exception as error:
        return ImageContextResult(
            payload=original_payload,
            original_payload=original_payload,
            mode=mode,
            fallback_stage="render",
            fallback_reason=_exception_reason("image context rendering failed", error),
        )

    return ImageContextResult(
        payload=transformed_payload,
        original_payload=original_payload,
        mode=mode,
        converted_parts=converted_parts,
    )
