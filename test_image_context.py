import base64
import io

from PIL import Image, ImageDraw

from src.converter import image_context
from src.converter.image_context import (
    IMAGE_CONTEXT_FONT_ENV,
    IMAGE_CONTEXT_MODE_FLAG,
    ImageContextMode,
    prepare_image_context_request,
)


def _assert_png_part(part):
    assert set(part) == {"inlineData"}
    inline_data = part["inlineData"]
    assert inline_data["mimeType"] == "image/png"
    png = base64.b64decode(inline_data["data"], validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(png)) as rendered:
        assert rendered.format == "PNG"
        assert rendered.width == image_context._STYLE.width
        assert rendered.height > 0


def test_all_text_mode_converts_contents_and_system_without_mutating_source():
    body = {
        "model": "gemini-test",
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.ALL_TEXT.value,
        "request": {
            "systemInstruction": {"parts": [{"text": "系统规则"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "old question"}]},
                {"role": "model", "parts": [{"text": "old answer"}]},
                {"role": "user", "parts": [{"text": "new question"}]},
            ],
        },
    }

    result = prepare_image_context_request(body)

    assert result.mode is ImageContextMode.ALL_TEXT
    assert result.converted_parts == 4
    assert result.applied
    assert not result.used_fallback
    assert result.original_payload is not None
    assert IMAGE_CONTEXT_MODE_FLAG not in result.payload
    assert IMAGE_CONTEXT_MODE_FLAG not in result.original_payload
    assert result.payload is not result.original_payload

    # The caller's object still contains its control flag and original text.
    assert body[IMAGE_CONTEXT_MODE_FLAG] == ImageContextMode.ALL_TEXT.value
    assert body["request"]["contents"][0]["parts"][0] == {"text": "old question"}
    assert result.original_payload["request"]["contents"][0]["parts"][0] == {"text": "old question"}

    _assert_png_part(result.payload["request"]["systemInstruction"]["parts"][0])
    for content in result.payload["request"]["contents"]:
        _assert_png_part(content["parts"][0])


def test_current_turn_starts_after_last_model_and_keeps_system_text():
    body = {
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.CURRENT_TURN.value,
        "request": {
            "systemInstruction": {"parts": [{"text": "keep system"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "first question"}]},
                {"role": "model", "parts": [{"text": "first answer"}]},
                {"role": "user", "parts": [{"text": "tool follow-up"}]},
                {"role": "model", "parts": [{"text": "latest answer"}]},
                {"role": "user", "parts": [{"text": "current question"}]},
            ],
        },
    }

    result = prepare_image_context_request(body)
    request = result.payload["request"]

    assert result.converted_parts == 1
    assert request["systemInstruction"]["parts"] == [{"text": "keep system"}]
    assert request["contents"][0]["parts"] == [{"text": "first question"}]
    assert request["contents"][1]["parts"] == [{"text": "first answer"}]
    assert request["contents"][2]["parts"] == [{"text": "tool follow-up"}]
    assert request["contents"][3]["parts"] == [{"text": "latest answer"}]
    _assert_png_part(request["contents"][4]["parts"][0])


def test_current_turn_without_model_history_converts_every_content():
    body = {
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.CURRENT_TURN.value,
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": "one"}]},
                {"role": "user", "parts": [{"text": "two"}]},
            ]
        },
    }

    result = prepare_image_context_request(body)

    assert result.converted_parts == 2
    for content in result.payload["request"]["contents"]:
        _assert_png_part(content["parts"][0])


def test_native_images_tools_empty_and_non_text_parts_are_preserved():
    native_image = {"inlineData": {"mimeType": "image/jpeg", "data": "native"}}
    file_data = {"fileData": {"mimeType": "image/png", "fileUri": "gs://bucket/file"}}
    function_call = {"functionCall": {"name": "lookup", "args": {"id": 1}}}
    function_response = {"functionResponse": {"name": "lookup", "response": {"name": "value"}}}
    executable_code = {"executableCode": {"language": "PYTHON", "code": "print(1)"}}
    empty_parts = [{"text": None}, {"text": ""}, {"text": " \n\t"}, {}]
    body = {
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.ALL_TEXT.value,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        native_image,
                        file_data,
                        function_call,
                        function_response,
                        executable_code,
                        *empty_parts,
                    ],
                }
            ]
        },
    }

    result = prepare_image_context_request(body)

    assert result.converted_parts == 0
    assert result.payload["request"]["contents"][0]["parts"] == [
        native_image,
        file_data,
        function_call,
        function_response,
        executable_code,
        *empty_parts,
    ]


def test_malformed_part_with_text_and_tool_data_is_not_destroyed():
    mixed_part = {"text": "do not replace", "functionCall": {"name": "tool", "args": {}}}
    body = {
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.ALL_TEXT.value,
        "request": {"contents": [{"role": "user", "parts": [mixed_part]}]},
    }

    result = prepare_image_context_request(body)

    assert result.converted_parts == 0
    assert result.payload["request"]["contents"][0]["parts"][0] == mixed_part


def test_pixel_wrapping_preserves_unicode_spaces_and_tab_indentation():
    source = "函数\tvalue = 1  # café e\u0301 🙂"
    measuring_image = Image.new("RGB", (400, 1), "white")
    draw = ImageDraw.Draw(measuring_image)
    font = image_context._load_font(24)
    max_width = 150

    wrapped = image_context._wrap_line_by_pixel_width(
        source,
        draw=draw,
        font=font,
        max_width=max_width,
    )

    assert len(wrapped) > 1
    assert "".join(wrapped) == source.expandtabs(image_context._STYLE.tab_size)
    assert all(image_context._pixel_width(draw, line, font) <= max_width for line in wrapped)
    # The combining accent remains attached to its base character.
    assert all(not line.startswith("\u0301") for line in wrapped)


def test_renderer_receives_role_and_original_part_index(monkeypatch):
    calls = []

    def fake_renderer(text, *, role, part_index):
        calls.append((text, role, part_index))
        return [b"png"]

    monkeypatch.setattr(image_context, "_render_text_png_pages", fake_renderer)
    body = {
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.ALL_TEXT.value,
        "request": {
            "contents": [{"role": "user", "parts": [{"functionCall": {}}, {"text": "code"}]}]
        },
    }

    result = prepare_image_context_request(body)

    assert result.converted_parts == 1
    assert calls == [("code", "user", 1)]


def test_long_text_is_split_into_unbounded_readable_png_pages():
    line_count = (image_context._STYLE.max_lines_per_page * 2) + 7
    text = "\n".join(f"line {index}" for index in range(line_count))
    body = {
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.ALL_TEXT.value,
        "request": {"contents": [{"role": "user", "parts": [{"text": text}]}]},
    }

    result = prepare_image_context_request(body)
    image_parts = result.payload["request"]["contents"][0]["parts"]

    assert result.converted_parts == 1
    assert len(image_parts) == 3
    for part in image_parts:
        _assert_png_part(part)
    assert image_context._page_header("user", 0, 0, 3) == ("role=user | part=1 | page=1/3")
    assert image_context._page_header("user", 0, 2, 3) == ("role=user | part=1 | page=3/3")


def test_paginated_layout_reconstructs_every_source_line_and_empty_line():
    source_lines = [
        "first",
        "",
        *[f"short line {index}" for index in range(image_context._STYLE.max_lines_per_page)],
        "",
        "last",
        "",
    ]
    source = "\n".join(source_lines)
    measuring_image = Image.new("RGB", (image_context._STYLE.width, 1), "white")
    draw = ImageDraw.Draw(measuring_image)
    font = image_context._load_font(image_context._STYLE.body_font_size)

    lines = image_context._layout_text_lines(
        source,
        draw=draw,
        font=font,
        max_width=image_context._STYLE.width - (2 * image_context._STYLE.horizontal_padding),
    )
    pages = image_context._paginate_lines(lines, image_context._STYLE.max_lines_per_page)

    assert len(pages) == 2
    assert [line for page in pages for line in page] == source_lines
    assert pages[0][1] == ""
    assert pages[-1][-1] == ""


def test_paginated_text_keeps_order_between_native_image_and_tool_parts(monkeypatch):
    function_call = {"functionCall": {"name": "before", "args": {}}}
    native_image = {"inlineData": {"mimeType": "image/jpeg", "data": "native"}}
    function_response = {"functionResponse": {"name": "after", "response": {}}}

    def two_page_renderer(text, *, role, part_index):
        assert (text, role, part_index) == ("long text", "user", 1)
        return [b"page one", b"page two"]

    monkeypatch.setattr(image_context, "_render_text_png_pages", two_page_renderer)
    body = {
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.ALL_TEXT.value,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        function_call,
                        {"text": "long text"},
                        native_image,
                        function_response,
                    ],
                }
            ]
        },
    }

    result = prepare_image_context_request(body)
    parts = result.payload["request"]["contents"][0]["parts"]

    assert result.converted_parts == 1
    assert parts[0] == function_call
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == b"page one"
    assert base64.b64decode(parts[2]["inlineData"]["data"]) == b"page two"
    assert parts[3] == native_image
    assert parts[4] == function_response


def test_render_failure_returns_the_complete_untransformed_request(monkeypatch):
    calls = 0

    def failing_renderer(text, *, role, part_index):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("font backend unavailable")
        return [b"first image"]

    monkeypatch.setattr(image_context, "_render_text_png_pages", failing_renderer)
    body = {
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.ALL_TEXT.value,
        "request": {"contents": [{"role": "user", "parts": [{"text": "one"}, {"text": "two"}]}]},
    }

    result = prepare_image_context_request(body)

    assert result.used_fallback
    assert result.fallback_stage == "render"
    assert result.fallback_reason == (
        "image context rendering failed: RuntimeError: font backend unavailable"
    )
    assert result.converted_parts == 0
    assert result.payload is result.original_payload
    assert IMAGE_CONTEXT_MODE_FLAG not in result.payload
    assert result.payload["request"]["contents"][0]["parts"] == [
        {"text": "one"},
        {"text": "two"},
    ]


def test_invalid_mode_is_consumed_and_returns_clear_fallback():
    body = {
        IMAGE_CONTEXT_MODE_FLAG: "future_mode",
        "request": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
    }

    result = prepare_image_context_request(body)

    assert result.mode is None
    assert result.used_fallback
    assert result.fallback_stage == "mode_validation"
    assert "future_mode" in result.fallback_reason
    assert result.payload is result.original_payload
    assert IMAGE_CONTEXT_MODE_FLAG not in result.payload
    assert IMAGE_CONTEXT_MODE_FLAG in body


def test_non_text_model_consumes_mode_without_transforming_prompt():
    body = {
        "model": "gemini-3.1-flash-image",
        IMAGE_CONTEXT_MODE_FLAG: ImageContextMode.ALL_TEXT.value,
        "request": {"contents": [{"role": "user", "parts": [{"text": "draw a cat"}]}]},
    }

    result = prepare_image_context_request(body)

    assert result.fallback_stage == "model_filter"
    assert result.converted_parts == 0
    assert result.payload["request"]["contents"][0]["parts"] == [{"text": "draw a cat"}]
    assert IMAGE_CONTEXT_MODE_FLAG not in result.payload


def test_absent_mode_is_a_zero_copy_noop():
    body = {"model": "gemini-test", "request": {"contents": []}}

    result = prepare_image_context_request(body)

    assert result.payload is body
    assert result.original_payload is None
    assert result.mode is None
    assert result.converted_parts == 0
    assert not result.applied
    assert not result.used_fallback


def test_font_environment_override_is_the_first_candidate(monkeypatch):
    monkeypatch.setenv(IMAGE_CONTEXT_FONT_ENV, "C:/custom/font.ttc")

    assert next(iter(image_context._font_candidates())) == "C:/custom/font.ttc"
