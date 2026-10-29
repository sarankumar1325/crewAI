"""Chat Completions streaming must not drop choices on chunks that carry usage.

Some OpenAI-compatible providers (for example DeepSeek) send ``usage`` on the
terminal chunk together with a choice that holds the last delta and the
``finish_reason``. Usage has to be recorded without skipping that choice.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from crewai.llms.providers.openai.completion import OpenAICompletion


USAGE = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def _chunk(
    *,
    content: str | None = None,
    tool_call: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: bool = False,
    no_choices: bool = False,
) -> ChatCompletionChunk:
    delta: dict[str, Any] = {"content": content}
    if tool_call is not None:
        delta["tool_calls"] = [tool_call]
    choices = (
        []
        if no_choices
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    )
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": choices,
            "usage": USAGE if usage else None,
        }
    )


def _sync_client(chunks: list[ChatCompletionChunk]) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: iter(chunks))
        )
    )


def _async_client(chunks: list[ChatCompletionChunk]) -> SimpleNamespace:
    async def _stream():
        for chunk in chunks:
            yield chunk

    async def _create(**_: Any):
        return _stream()

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )


TEXT_CASES = {
    "last_text_delta_with_usage": [
        _chunk(content="HELLO-"),
        _chunk(content="END", finish_reason="stop", usage=True),
    ],
    "only_chunk_with_usage": [
        _chunk(content="HELLO-END", finish_reason="stop", usage=True),
    ],
    "empty_delta_with_usage_and_finish_reason": [
        _chunk(content="HELLO-END"),
        _chunk(content="", finish_reason="stop", usage=True),
    ],
    "separate_usage_only_chunk": [
        _chunk(content="HELLO-END", finish_reason="stop"),
        _chunk(usage=True, no_choices=True),
    ],
}


def _llm() -> OpenAICompletion:
    return OpenAICompletion(model="gpt-4o-mini", api_key="unused", stream=True)


def _assert_usage_recorded_once(llm: OpenAICompletion) -> None:
    summary = llm.get_token_usage_summary()
    assert summary.successful_requests == 1
    assert summary.total_tokens == USAGE["total_tokens"]


@pytest.mark.parametrize("chunks", TEXT_CASES.values(), ids=TEXT_CASES.keys())
def test_sync_stream_keeps_choice_on_usage_chunk(chunks):
    llm = _llm()
    with (
        patch.object(llm, "_get_sync_client", return_value=_sync_client(chunks)),
        patch.object(llm, "_emit_call_completed_event") as completed,
    ):
        result = llm._handle_streaming_completion({"messages": []})

    assert result == "HELLO-END"
    assert completed.call_args.kwargs["finish_reason"] == "stop"
    _assert_usage_recorded_once(llm)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", TEXT_CASES.values(), ids=TEXT_CASES.keys())
async def test_async_stream_keeps_choice_on_usage_chunk(chunks):
    llm = _llm()
    with (
        patch.object(llm, "_get_async_client", return_value=_async_client(chunks)),
        patch.object(llm, "_emit_call_completed_event") as completed,
    ):
        result = await llm._ahandle_streaming_completion({"messages": []})

    assert result == "HELLO-END"
    assert completed.call_args.kwargs["finish_reason"] == "stop"
    _assert_usage_recorded_once(llm)


TOOL_CHUNKS = [
    _chunk(
        tool_call={
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr'},
        }
    ),
    _chunk(
        tool_call={"index": 0, "function": {"arguments": 'ession": "1+1"}'}},
        finish_reason="tool_calls",
        usage=True,
    ),
]


def test_sync_stream_keeps_tool_call_suffix_on_usage_chunk():
    llm = _llm()
    with patch.object(
        llm, "_get_sync_client", return_value=_sync_client(TOOL_CHUNKS)
    ):
        result = llm._handle_streaming_completion({"messages": []})

    assert isinstance(result, list)
    assert result[0]["function"]["arguments"] == '{"expression": "1+1"}'


@pytest.mark.asyncio
async def test_async_stream_keeps_tool_call_suffix_on_usage_chunk():
    llm = _llm()
    with patch.object(
        llm, "_get_async_client", return_value=_async_client(TOOL_CHUNKS)
    ):
        result = await llm._ahandle_streaming_completion({"messages": []})

    assert isinstance(result, list)
    assert result[0]["function"]["arguments"] == '{"expression": "1+1"}'


class Answer(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_async_structured_stream_keeps_json_suffix_on_usage_chunk():
    llm = _llm()
    chunks = [
        _chunk(content='{"answer": '),
        _chunk(content='"42"}', finish_reason="stop", usage=True),
    ]
    with patch.object(llm, "_get_async_client", return_value=_async_client(chunks)):
        result = await llm._ahandle_streaming_completion(
            {"messages": []}, response_model=Answer
        )

    assert result == Answer(answer="42")
    _assert_usage_recorded_once(llm)
