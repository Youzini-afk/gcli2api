"""
GeminiCli API Client - Handles all communication with GeminiCli API.
This module is used by both OpenAI compatibility layer and native Gemini endpoints.
GeminiCli API 客户端 - 处理与 GeminiCli API 的所有通信
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径（用于直接运行测试）
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import asyncio
import json
from typing import Any, Dict, Optional, Callable, Tuple

from fastapi import Response
from config import get_code_assist_endpoint, get_auto_ban_error_codes
from log import log

from src.credential_manager import credential_manager
from src.httpx_client import stream_post_async, post_async

# 导入共同的基础功能
from src.api.utils import (
    handle_error_with_retry,
    get_retry_config,
    record_api_call_success,
    record_api_call_error,
    parse_and_log_cooldown,
)
from src.utils import get_geminicli_user_agent
from src.converter.tool_anti_truncation import (
    TOOL_ANTI_TRUNCATION_FLAG,
    prepare_tool_anti_truncation_request,
    transform_non_stream_response,
    transform_stream_item,
)
from src.converter.image_context import (
    IMAGE_CONTEXT_MODE_FLAG,
    prepare_image_context_request,
)

# ==================== 全局凭证管理器 ====================

# 使用全局单例 credential_manager，自动初始化


# ==================== 请求准备 ====================

async def prepare_request_headers_and_payload(
    payload: dict, credential_data: dict, target_url: str
):
    """
    从凭证数据准备请求头和最终payload
    
    Args:
        payload: 原始请求payload
        credential_data: 凭证数据字典
        target_url: 目标URL
        
    Returns:
        元组: (headers, final_payload, target_url)
        
    Raises:
        Exception: 如果凭证中缺少必要字段
    """
    token = credential_data.get("token") or credential_data.get("access_token", "")
    if not token:
        raise Exception("凭证中没有找到有效的访问令牌（token或access_token字段）")

    source_request = payload.get("request", {})

    # 内部API使用Bearer Token和项目ID
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": get_geminicli_user_agent(payload.get("model", "")),
    }
    project_id = credential_data.get("project_id", "")
    if not project_id:
        raise Exception("项目ID不存在于凭证数据中")
    final_payload = {
        "model": payload.get("model"),
        "project": project_id,
        "request": source_request,
    }

    return headers, final_payload, target_url


def _is_retryable_status(status_code: int, disable_error_codes: list[int]) -> bool:
    """统一判断是否属于可重试状态码。"""
    return status_code in (429, 500, 503) or status_code in disable_error_codes


_IMAGE_CONTEXT_ERROR_STATUSES = {400, 415, 422}
_IMAGE_CONTEXT_MARKERS = (
    "inlinedata",
    "inline_data",
    "inline data",
    "multimodal",
    "multi-modal",
    "image input",
    "image data",
    "image part",
    "image context",
    "image content",
    "image is",
    "support image",
    "image/",
)
_IMAGE_CONTEXT_REJECTION_MARKERS = (
    "invalid",
    "unsupported",
    "not supported",
    "unknown",
    "unrecognized",
    "malformed",
    "not allowed",
    "cannot",
    "can't",
    "expected",
    "must be",
    "format",
    "no such field",
    "not enabled",
    "not available",
)
_QUOTA_ERROR_MARKERS = (
    "quota",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "too many requests",
    "billing",
    "credit limit",
    "capacity",
)


def _is_confirmed_image_context_error(status_code: int, error_text: str) -> bool:
    """Return true only for an explicit upstream multimodal-shape rejection."""
    if status_code not in _IMAGE_CONTEXT_ERROR_STATUSES or not error_text:
        return False
    normalized = error_text.lower()
    if any(marker in normalized for marker in _QUOTA_ERROR_MARKERS):
        return False
    has_image_marker = any(marker in normalized for marker in _IMAGE_CONTEXT_MARKERS)
    return has_image_marker and any(
        marker in normalized for marker in _IMAGE_CONTEXT_REJECTION_MARKERS
    )


async def _prepare_request_variants(body: Dict[str, Any]):
    """Prepare image and text-fallback bodies before synthetic-tool injection."""
    if IMAGE_CONTEXT_MODE_FLAG not in body:
        prepared, context = prepare_tool_anti_truncation_request(body)
        return prepared, context, None, None, False

    image_result = await asyncio.to_thread(prepare_image_context_request, body)
    if not image_result.mode:
        prepared, context = prepare_tool_anti_truncation_request(image_result.payload)
        return prepared, context, None, None, False

    image_result.payload[TOOL_ANTI_TRUNCATION_FLAG] = True
    if not image_result.applied:
        prepared, context = prepare_tool_anti_truncation_request(image_result.payload)
        return prepared, context, None, None, False

    image_result.original_payload[TOOL_ANTI_TRUNCATION_FLAG] = True
    prepared, context = prepare_tool_anti_truncation_request(image_result.payload)
    fallback, fallback_context = prepare_tool_anti_truncation_request(
        image_result.original_payload
    )
    return prepared, context, fallback, fallback_context, bool(image_result.applied)


async def _switch_credential_for_retry(
    *,
    next_cred_task: Optional[asyncio.Task],
    retry_interval: float,
    refresh_credential_fast: Callable[[], Any],
    apply_cred_result: Callable[[Tuple[str, Dict[str, Any]]], bool],
    log_prefix: str,
) -> Tuple[bool, Optional[asyncio.Task]]:
    """优先使用预热凭证，失败后退回同步刷新。"""
    if next_cred_task is not None:
        try:
            cred_result = await next_cred_task
            next_cred_task = None
            if cred_result and apply_cred_result(cred_result):
                await asyncio.sleep(retry_interval)
                return True, next_cred_task
        except Exception as e:
            log.warning(f"{log_prefix} 预热凭证任务失败: {e}")
            next_cred_task = None

    await asyncio.sleep(retry_interval)
    if await refresh_credential_fast():
        return True, next_cred_task

    return False, next_cred_task


# ==================== 新的流式和非流式请求函数 ====================

async def stream_request(
    body: Dict[str, Any],
    native: bool = False,
    headers: Optional[Dict[str, str]] = None,
):
    """
    流式请求函数

    Args:
        body: 请求体
        native: 是否返回原生bytes流，False则返回str流
        headers: 额外的请求头

    Yields:
        Response对象（错误时）或 bytes流/str流（成功时）
    """
    (
        body,
        tool_anti_context,
        fallback_body,
        fallback_tool_anti_context,
        image_fallback_available,
    ) = await _prepare_request_variants(body)
    # 获取有效凭证
    model_name = body.get("model", "")

    # 1. 获取有效凭证
    cred_result = await credential_manager.get_valid_credential(
        mode="geminicli", model_name=model_name
    )

    if not cred_result:
        # 如果返回值是None，直接返回错误500
        yield Response(
            content=json.dumps({"error": "当前无可用凭证"}),
            status_code=500,
            media_type="application/json"
        )
        return

    current_file, credential_data = cred_result

    # 2. 构建URL和请求头
    try:
        auth_headers, final_payload, target_url = await prepare_request_headers_and_payload(
            body, credential_data,
            f"{await get_code_assist_endpoint()}/v1internal:streamGenerateContent?alt=sse"
        )
        fallback_final_payload = None
        if fallback_body is not None:
            _, fallback_final_payload, _ = await prepare_request_headers_and_payload(
                fallback_body, credential_data, target_url
            )

        # 合并自定义headers
        if headers:
            auth_headers.update(headers)

    except Exception as e:
        log.error(f"准备请求失败: {e}")
        yield Response(
            content=json.dumps({"error": f"准备请求失败: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )
        return

    use_text_fallback = False

    async def stream_current_payload():
        """Retry a confirmed multimodal rejection once, without changing credential."""
        nonlocal use_text_fallback
        request_payload = (
            fallback_final_payload
            if use_text_fallback and fallback_final_payload is not None
            else final_payload
        )
        emitted_data = False
        retry_with_text = False
        upstream_stream = stream_post_async(
            url=target_url,
            body=request_payload,
            native=native,
            headers=auth_headers,
        )
        async for item in upstream_stream:
            if (
                isinstance(item, Response)
                and image_fallback_available
                and not use_text_fallback
                and not emitted_data
            ):
                try:
                    item_error = (
                        item.body.decode("utf-8")
                        if isinstance(item.body, bytes)
                        else str(item.body)
                    )
                except Exception:
                    item_error = ""
                if _is_confirmed_image_context_error(item.status_code, item_error):
                    use_text_fallback = True
                    retry_with_text = True
                    log.info(
                        "[GEMINICLI STREAM] 上游拒绝图片上下文，"
                        f"使用同一凭证立即回退纯文本表示: {current_file}"
                    )
                    break
            if not isinstance(item, Response):
                emitted_data = True
            yield item
        if not retry_with_text:
            return

        close_stream = getattr(upstream_stream, "aclose", None)
        if close_stream is not None:
            await close_stream()
        async for fallback_item in stream_post_async(
            url=target_url,
            body=fallback_final_payload,
            native=native,
            headers=auth_headers,
        ):
            yield fallback_item

    # 3. 调用stream_post_async进行请求
    retry_config = await get_retry_config()
    max_retries = retry_config["max_retries"]
    retry_interval = retry_config["retry_interval"]

    DISABLE_ERROR_CODES = await get_auto_ban_error_codes()  # 禁用凭证的错误码
    last_error_response = None  # 记录最后一次的错误响应
    next_cred_task = None  # 预热的下一个凭证任务

    # 内部函数：快速更新凭证(只更新token和project_id,避免重建整个请求)
    async def refresh_credential_fast():
        nonlocal current_file, credential_data, auth_headers, final_payload
        cred_result = await credential_manager.get_valid_credential(
            mode="geminicli", model_name=model_name
        )
        if not cred_result:
            return None
        current_file, credential_data = cred_result
        try:
            # 只更新token和project_id,不重建整个headers和payload
            token = credential_data.get("token") or credential_data.get("access_token", "")
            project_id = credential_data.get("project_id", "")
            if not token or not project_id:
                return None

            # 直接更新现有的headers和payload
            auth_headers["Authorization"] = f"Bearer {token}"
            final_payload["project"] = project_id
            if fallback_final_payload is not None:
                fallback_final_payload["project"] = project_id
            return True
        except Exception:
            return None

    def apply_cred_result(cred_result: Tuple[str, Dict[str, Any]]) -> bool:
        nonlocal current_file, credential_data, auth_headers, final_payload
        current_file, credential_data = cred_result
        token = credential_data.get("token") or credential_data.get("access_token", "")
        project_id = credential_data.get("project_id", "")
        if not token or not project_id:
            return False
        auth_headers["Authorization"] = f"Bearer {token}"
        final_payload["project"] = project_id
        if fallback_final_payload is not None:
            fallback_final_payload["project"] = project_id
        return True

    for attempt in range(max_retries + 1):
        success_recorded = False  # 标记是否已记录成功
        need_retry = False  # 标记是否需要重试

        try:
            async for chunk in stream_current_payload():
                # 判断是否是Response对象
                if isinstance(chunk, Response):
                    status_code = chunk.status_code
                    last_error_response = chunk  # 记录最后一次错误

                    if success_recorded:
                        log.warning(
                            "[GEMINICLI STREAM] 已输出部分响应后收到上游错误，"
                            "原样返回错误并停止，避免重试造成重复内容"
                        )
                        yield chunk
                        return

                    # 缓存错误解析结果,避免重复decode
                    error_body = None
                    try:
                        error_body = chunk.body.decode('utf-8') if isinstance(chunk.body, bytes) else str(chunk.body)
                    except Exception:
                        error_body = ""

                    # 如果错误码是429、503或者在禁用码当中，做好记录后进行重试
                    if _is_retryable_status(status_code, DISABLE_ERROR_CODES):
                        log.warning(f"[GEMINICLI STREAM] 流式请求失败 (status={status_code}), 凭证: {current_file}, 响应: {error_body[:500] if error_body else '无'}")

                        # 解析冷却时间
                        cooldown_until = None
                        if (status_code == 429 or status_code == 503) and error_body:
                            try:
                                cooldown_until = await parse_and_log_cooldown(error_body, mode="geminicli")
                            except Exception:
                                pass

                        # 预热下一个凭证
                        if next_cred_task is None and attempt < max_retries:
                            next_cred_task = asyncio.create_task(
                                credential_manager.get_valid_credential(
                                    mode="geminicli", model_name=model_name
                                )
                            )

                        # 记录错误并切换凭证
                        await record_api_call_error(
                            credential_manager, current_file, status_code,
                            cooldown_until, mode="geminicli", model_name=model_name,
                            error_message=error_body
                        )

                        # 检查是否应该重试
                        should_retry = await handle_error_with_retry(
                            credential_manager, status_code, current_file,
                            retry_config["retry_enabled"], attempt, max_retries, retry_interval,
                            mode="geminicli"
                        )

                        if should_retry and attempt < max_retries:
                            need_retry = True
                            break  # 跳出内层循环，准备重试
                        else:
                            # 不重试，直接返回原始错误
                            log.error(f"[GEMINICLI STREAM] 达到最大重试次数或不应重试，返回原始错误")
                            yield chunk
                            return
                    elif status_code == 404 and "preview" in model_name.lower():
                        # 特殊处理：preview模型返回404，说明该凭证不支持preview模型
                        log.warning(f"[GEMINICLI STREAM] Preview模型404错误，凭证不支持preview: {current_file}")

                        # 将该凭证的preview状态设置为False
                        try:
                            await credential_manager.update_credential_state(
                                current_file, {"preview": False}, mode="geminicli"
                            )
                            log.info(f"[GEMINICLI STREAM] 已将凭证 {current_file} 的preview状态设置为False")
                        except Exception as e:
                            log.error(f"[GEMINICLI STREAM] 更新凭证preview状态失败: {e}")

                        # 记录404错误
                        await record_api_call_error(
                            credential_manager, current_file, status_code,
                            None, mode="geminicli", model_name=model_name,
                            error_message=error_body
                        )

                        # 预热下一个凭证（会自动跳过preview=False的凭证）
                        if next_cred_task is None and attempt < max_retries:
                            next_cred_task = asyncio.create_task(
                                credential_manager.get_valid_credential(
                                    mode="geminicli", model_name=model_name
                                )
                            )

                        # 触发重试
                        if attempt < max_retries:
                            need_retry = True
                            break
                        else:
                            log.error(f"[GEMINICLI STREAM] 达到最大重试次数，返回404错误")
                            yield chunk
                            return
                    else:
                        # 错误码不在禁用码当中，直接返回，无需重试
                        log.error(f"[GEMINICLI STREAM] 流式请求失败，非重试错误码 (status={status_code}), 凭证: {current_file}, 响应: {error_body[:500] if error_body else '无'}")
                        await record_api_call_error(
                            credential_manager, current_file, status_code,
                            None, mode="geminicli", model_name=model_name,
                            error_message=error_body
                        )
                        yield chunk
                        return
                else:
                    # 不是Response，说明是真流，直接yield返回
                    # 只在第一个chunk时记录成功
                    if not success_recorded:
                        await record_api_call_success(
                            credential_manager, current_file, mode="geminicli", model_name=model_name
                        )
                        success_recorded = True
                        log.debug(f"[GEMINICLI STREAM] 开始接收流式响应，模型: {model_name}")

                    active_tool_context = (
                        fallback_tool_anti_context
                        if use_text_fallback
                        else tool_anti_context
                    )
                    yield transform_stream_item(chunk, active_tool_context)

            # 流式请求完成，检查结果
            if success_recorded:
                log.debug(f"[GEMINICLI STREAM] 流式响应完成，模型: {model_name}")
                return

            # 统一处理重试
            if need_retry:
                # 如果已经是最后一次尝试，不再重试，直接返回错误
                if attempt >= max_retries:
                    log.error(f"[GEMINICLI STREAM] 达到最大重试次数，返回错误")
                    if last_error_response:
                        yield last_error_response
                    else:
                        yield Response(
                            content=json.dumps({"error": "请求失败，所有重试均已耗尽"}),
                            status_code=429,
                            media_type="application/json"
                        )
                    return

                log.info(f"[GEMINICLI STREAM] 重试请求 (attempt {attempt + 2}/{max_retries + 1})...")

                switched, next_cred_task = await _switch_credential_for_retry(
                    next_cred_task=next_cred_task,
                    retry_interval=retry_interval,
                    refresh_credential_fast=refresh_credential_fast,
                    apply_cred_result=apply_cred_result,
                    log_prefix="[GEMINICLI STREAM]",
                )
                if not switched:
                    log.error("[GEMINICLI STREAM] 重试时无可用凭证或刷新失败")
                    yield Response(
                        content=json.dumps({"error": "当前无可用凭证"}),
                        status_code=500,
                        media_type="application/json"
                    )
                    return
                continue  # 重试

        except Exception as e:
            log.error(f"[GEMINICLI STREAM] 流式请求异常: {e}, 凭证: {current_file}")
            if success_recorded:
                log.warning("[GEMINICLI STREAM] 已输出部分响应，不再重试以避免重复内容")
                yield Response(
                    content=json.dumps(
                        {
                            "error": {
                                "code": 502,
                                "message": "上游流在输出部分内容后中断",
                                "status": "UPSTREAM_STREAM_INTERRUPTED",
                            }
                        },
                        ensure_ascii=False,
                    ),
                    status_code=502,
                    media_type="application/json",
                )
                return
            if attempt < max_retries:
                log.info(f"[GEMINICLI STREAM] 异常后重试 (attempt {attempt + 2}/{max_retries + 1})...")
                await asyncio.sleep(retry_interval)
                continue
            else:
                # 所有重试都失败，返回最后一次的错误（如果有）
                log.error(f"[GEMINICLI STREAM] 所有重试均失败，最后异常: {e}")
                if last_error_response:
                    yield last_error_response
                else:
                    # 如果没有记录到错误响应，返回500错误
                    yield Response(
                        content=json.dumps({"error": f"流式请求异常: {str(e)}"}),
                        status_code=500,
                        media_type="application/json"
                    )
                return

    # 所有重试均已耗尽（for循环正常结束），返回最后记录的错误
    log.error("[GEMINICLI STREAM] 所有重试均失败")
    if last_error_response:
        yield last_error_response
    else:
        yield Response(
            content=json.dumps({"error": "请求失败，所有重试均已耗尽"}),
            status_code=429,
            media_type="application/json"
        )


async def non_stream_request(
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """
    非流式请求函数

    Args:
        body: 请求体
        native: 保留参数以保持接口一致性（实际未使用）
        headers: 额外的请求头

    Returns:
        Response对象
    """
    (
        body,
        tool_anti_context,
        fallback_body,
        fallback_tool_anti_context,
        image_fallback_available,
    ) = await _prepare_request_variants(body)
    # 获取有效凭证
    model_name = body.get("model", "")

    # 1. 获取有效凭证
    cred_result = await credential_manager.get_valid_credential(
        mode="geminicli", model_name=model_name
    )

    if not cred_result:
        # 如果返回值是None，直接返回错误500
        return Response(
            content=json.dumps({"error": "当前无可用凭证"}),
            status_code=500,
            media_type="application/json"
        )

    current_file, credential_data = cred_result

    # 2. 构建URL和请求头
    try:
        auth_headers, final_payload, target_url = await prepare_request_headers_and_payload(
            body, credential_data,
            f"{await get_code_assist_endpoint()}/v1internal:generateContent"
        )
        fallback_final_payload = None
        if fallback_body is not None:
            _, fallback_final_payload, _ = await prepare_request_headers_and_payload(
                fallback_body, credential_data, target_url
            )

        # 合并自定义headers
        if headers:
            auth_headers.update(headers)

    except Exception as e:
        log.error(f"准备请求失败: {e}")
        return Response(
            content=json.dumps({"error": f"准备请求失败: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )

    use_text_fallback = False

    async def post_current_payload():
        """Retry a confirmed multimodal rejection once, without changing credential."""
        nonlocal use_text_fallback
        request_payload = (
            fallback_final_payload
            if use_text_fallback and fallback_final_payload is not None
            else final_payload
        )
        response = await post_async(
            url=target_url, json=request_payload, headers=auth_headers
        )
        if image_fallback_available and not use_text_fallback:
            try:
                response_error = response.text
            except Exception:
                response_error = ""
            if _is_confirmed_image_context_error(response.status_code, response_error):
                use_text_fallback = True
                log.info(
                    "[GEMINICLI] 上游拒绝图片上下文，"
                    f"使用同一凭证立即回退纯文本表示: {current_file}"
                )
                return await post_async(
                    url=target_url,
                    json=fallback_final_payload,
                    headers=auth_headers,
                )
        return response

    # 3. 调用post_async进行请求
    retry_config = await get_retry_config()
    max_retries = retry_config["max_retries"]
    retry_interval = retry_config["retry_interval"]

    DISABLE_ERROR_CODES = await get_auto_ban_error_codes()  # 禁用凭证的错误码
    last_error_response = None  # 记录最后一次的错误响应
    next_cred_task = None  # 预热的下一个凭证任务

    # 内部函数：快速更新凭证(只更新token和project_id,避免重建整个请求)
    async def refresh_credential_fast():
        nonlocal current_file, credential_data, auth_headers, final_payload
        cred_result = await credential_manager.get_valid_credential(
            mode="geminicli", model_name=model_name
        )
        if not cred_result:
            return None
        current_file, credential_data = cred_result
        try:
            # 只更新token和project_id,不重建整个headers和payload
            token = credential_data.get("token") or credential_data.get("access_token", "")
            project_id = credential_data.get("project_id", "")
            if not token or not project_id:
                return None

            # 直接更新现有的headers和payload
            auth_headers["Authorization"] = f"Bearer {token}"
            final_payload["project"] = project_id
            if fallback_final_payload is not None:
                fallback_final_payload["project"] = project_id
            return True
        except Exception:
            return None

    def apply_cred_result(cred_result: Tuple[str, Dict[str, Any]]) -> bool:
        nonlocal current_file, credential_data, auth_headers, final_payload
        current_file, credential_data = cred_result
        token = credential_data.get("token") or credential_data.get("access_token", "")
        project_id = credential_data.get("project_id", "")
        if not token or not project_id:
            return False
        auth_headers["Authorization"] = f"Bearer {token}"
        final_payload["project"] = project_id
        if fallback_final_payload is not None:
            fallback_final_payload["project"] = project_id
        return True

    for attempt in range(max_retries + 1):
        try:
            response = await post_current_payload()

            status_code = response.status_code

            # 成功
            if status_code == 200:
                await record_api_call_success(
                    credential_manager, current_file, mode="geminicli", model_name=model_name
                )
                # 创建响应头,移除压缩相关的header避免重复解压
                response_headers = dict(response.headers)
                response_headers.pop('content-encoding', None)
                response_headers.pop('content-length', None)

                return Response(
                    content=transform_non_stream_response(
                        response.content,
                        fallback_tool_anti_context
                        if use_text_fallback
                        else tool_anti_context,
                    ),
                    status_code=200,
                    headers=response_headers
                )

            # 失败 - 记录最后一次错误
            # 创建响应头,移除压缩相关的header避免重复解压
            error_headers = dict(response.headers)
            error_headers.pop('content-encoding', None)
            error_headers.pop('content-length', None)

            last_error_response = Response(
                content=response.content,
                status_code=status_code,
                headers=error_headers
            )

            # 判断是否需要重试
            # 缓存错误文本,避免重复解析
            error_text = ""
            try:
                error_text = response.text
            except Exception:
                pass

            # 统一处理所有需要重试的错误码（429、503、禁用码）
            if _is_retryable_status(status_code, DISABLE_ERROR_CODES):
                log.warning(f"[NON-STREAM] 非流式请求失败 (status={status_code}), 凭证: {current_file}, 响应: {error_text[:500] if error_text else '无'}")

                # 解析冷却时间
                cooldown_until = None
                if (status_code == 429 or status_code == 503) and error_text:
                    try:
                        cooldown_until = await parse_and_log_cooldown(error_text, mode="geminicli")
                    except Exception:
                        pass

                # 并行预热下一个凭证,不阻塞当前处理
                if next_cred_task is None and attempt < max_retries:
                    next_cred_task = asyncio.create_task(
                        credential_manager.get_valid_credential(
                            mode="geminicli", model_name=model_name
                        )
                    )

                # 记录错误并切换凭证
                await record_api_call_error(
                    credential_manager, current_file, status_code,
                    cooldown_until, mode="geminicli", model_name=model_name,
                    error_message=error_text
                )

                # 检查是否应该重试（会自动处理禁用逻辑）
                should_retry = await handle_error_with_retry(
                    credential_manager, status_code, current_file,
                    retry_config["retry_enabled"], attempt, max_retries, retry_interval,
                    mode="geminicli"
                )

                if should_retry and attempt < max_retries:
                    # 重新获取凭证并重试
                    log.info(f"[NON-STREAM] 重试请求 (attempt {attempt + 2}/{max_retries + 1})...")

                    switched, next_cred_task = await _switch_credential_for_retry(
                        next_cred_task=next_cred_task,
                        retry_interval=retry_interval,
                        refresh_credential_fast=refresh_credential_fast,
                        apply_cred_result=apply_cred_result,
                        log_prefix="[NON-STREAM]",
                    )
                    if not switched:
                        log.error("[NON-STREAM] 重试时无可用凭证或刷新失败")
                        return Response(
                            content=json.dumps({"error": "当前无可用凭证"}),
                            status_code=500,
                            media_type="application/json"
                        )
                    continue  # 重试
                else:
                    # 不重试，直接返回原始错误
                    log.error(f"[NON-STREAM] 达到最大重试次数或不应重试，返回原始错误")
                    return last_error_response
            elif status_code == 404 and "preview" in model_name.lower():
                # 特殊处理：preview模型返回404，说明该凭证不支持preview模型
                log.warning(f"[NON-STREAM] Preview模型404错误，凭证不支持preview: {current_file}")

                # 将该凭证的preview状态设置为False
                try:
                    await credential_manager.update_credential_state(
                        current_file, {"preview": False}, mode="geminicli"
                    )
                    log.info(f"[NON-STREAM] 已将凭证 {current_file} 的preview状态设置为False")
                except Exception as e:
                    log.error(f"[NON-STREAM] 更新凭证preview状态失败: {e}")

                # 记录404错误
                await record_api_call_error(
                    credential_manager, current_file, status_code,
                    None, mode="geminicli", model_name=model_name,
                    error_message=error_text
                )

                # 预热下一个凭证（会自动跳过preview=False的凭证）
                if next_cred_task is None and attempt < max_retries:
                    next_cred_task = asyncio.create_task(
                        credential_manager.get_valid_credential(
                            mode="geminicli", model_name=model_name
                        )
                    )

                # 触发重试
                if attempt < max_retries:
                    log.info(f"[NON-STREAM] 重试请求 (attempt {attempt + 2}/{max_retries + 1})...")

                    switched, next_cred_task = await _switch_credential_for_retry(
                        next_cred_task=next_cred_task,
                        retry_interval=retry_interval,
                        refresh_credential_fast=refresh_credential_fast,
                        apply_cred_result=apply_cred_result,
                        log_prefix="[NON-STREAM]",
                    )
                    if not switched:
                        log.error("[NON-STREAM] 重试时无可用凭证或刷新失败")
                        return Response(
                            content=json.dumps({"error": "当前无可用凭证"}),
                            status_code=500,
                            media_type="application/json"
                        )
                    continue  # 重试
                else:
                    log.error(f"[NON-STREAM] 达到最大重试次数，返回404错误")
                    return last_error_response
            else:
                # 错误码不在重试范围内，直接返回
                log.error(f"[NON-STREAM] 非流式请求失败，非重试错误码 (status={status_code}), 凭证: {current_file}, 响应: {error_text[:500] if error_text else '无'}")
                await record_api_call_error(
                    credential_manager, current_file, status_code,
                    None, mode="geminicli", model_name=model_name,
                    error_message=error_text
                )
                return last_error_response

        except Exception as e:
            log.error(f"非流式请求异常: {e}, 凭证: {current_file}")
            if attempt < max_retries:
                log.info(f"[NON-STREAM] 异常后重试 (attempt {attempt + 2}/{max_retries + 1})...")
                await asyncio.sleep(retry_interval)
                continue
            else:
                # 所有重试都失败，返回最后一次的错误（如果有）或500错误
                log.error(f"[NON-STREAM] 所有重试均失败，最后异常: {e}")
                if last_error_response:
                    return last_error_response
                else:
                    return Response(
                        content=json.dumps({"error": f"请求异常: {str(e)}"}),
                        status_code=500,
                        media_type="application/json"
                    )

    # 所有重试都失败，返回最后一次的原始错误
    log.error("[NON-STREAM] 所有重试均失败")
    return last_error_response


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """
    测试代码：演示API返回的流式和非流式数据格式
    运行方式: python src/api/geminicli.py
    """
    print("=" * 80)
    print("GeminiCli API 测试")
    print("=" * 80)

    # 测试请求体
    test_body = {
        "model": "gemini-2.5-flash",
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Hello, tell me a joke in one sentence."}]
                }
            ]
        }
    }

    async def test_stream_request():
        """测试流式请求"""
        print("\n" + "=" * 80)
        print("【测试1】流式请求 (stream_request with native=False)")
        print("=" * 80)
        print(f"请求体: {json.dumps(test_body, indent=2, ensure_ascii=False)}\n")

        print("流式响应数据 (每个chunk):")
        print("-" * 80)

        chunk_count = 0
        async for chunk in stream_request(body=test_body, native=False):
            chunk_count += 1
            if isinstance(chunk, Response):
                # 错误响应
                print(f"\n❌ 错误响应:")
                print(f"  状态码: {chunk.status_code}")
                print(f"  Content-Type: {chunk.headers.get('content-type', 'N/A')}")
                try:
                    content = chunk.body.decode('utf-8') if isinstance(chunk.body, bytes) else str(chunk.body)
                    print(f"  内容: {content}")
                except Exception as e:
                    print(f"  内容解析失败: {e}")
            else:
                # 正常的流式数据块 (str类型)
                print(f"\nChunk #{chunk_count}:")
                print(f"  类型: {type(chunk).__name__}")
                print(f"  长度: {len(chunk) if hasattr(chunk, '__len__') else 'N/A'}")
                print(f"  内容预览: {repr(chunk[:200] if len(chunk) > 200 else chunk)}")

                # 如果是SSE格式，尝试解析
                if isinstance(chunk, str) and chunk.startswith("data: "):
                    try:
                        data_line = chunk.strip()
                        if data_line.startswith("data: "):
                            json_str = data_line[6:]  # 去掉 "data: " 前缀
                            json_data = json.loads(json_str)
                            print(f"  解析后的JSON: {json.dumps(json_data, indent=4, ensure_ascii=False)}")
                    except Exception as e:
                        print(f"  SSE解析尝试失败: {e}")

        print(f"\n总共收到 {chunk_count} 个chunk")

    async def test_non_stream_request():
        """测试非流式请求"""
        print("\n" + "=" * 80)
        print("【测试2】非流式请求 (non_stream_request)")
        print("=" * 80)
        print(f"请求体: {json.dumps(test_body, indent=2, ensure_ascii=False)}\n")

        response = await non_stream_request(body=test_body)

        print("非流式响应数据:")
        print("-" * 80)
        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'N/A')}")
        print(f"\n响应头: {dict(response.headers)}\n")

        try:
            content = response.body.decode('utf-8') if isinstance(response.body, bytes) else str(response.body)
            print(f"响应内容 (原始):\n{content}\n")

            # 尝试解析JSON
            try:
                json_data = json.loads(content)
                print(f"响应内容 (格式化JSON):")
                print(json.dumps(json_data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("(非JSON格式)")
        except Exception as e:
            print(f"内容解析失败: {e}")

    async def main():
        """主测试函数"""
        try:
            # 测试流式请求
            await test_stream_request()

            # 测试非流式请求
            await test_non_stream_request()

            print("\n" + "=" * 80)
            print("测试完成")
            print("=" * 80)

        except Exception as e:
            print(f"\n❌ 测试过程中出现异常: {e}")
            import traceback
            traceback.print_exc()

    # 运行测试
    asyncio.run(main())
