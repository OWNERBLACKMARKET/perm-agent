from __future__ import annotations

import json
from typing import Any

import litellm


class AsyncLlmHandler:
    """Async version of LlmHandler using litellm.acompletion."""

    async def execute(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = {"type": "json_object"}

        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content

        if response_format and content:
            content = json.loads(content)

        return content
