import asyncio
import re
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError, ParamValidationError
from prompts.prompt import (RECALL_CARD_SYSTEM_PROMPT,
                            SONG_AUGMENTED_QUERY_SYSTEM_PROMPT,
                            VIBE_CARD_SYSTEM_PROMPT)
from tqdm.asyncio import tqdm

_LRC_TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}\.\d{3}\]\s*")
_RECALL_CARD_TOOL_NAME = "recall_card_extractor"
_RECALL_CARD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "Noun_Key_Words": {"type": "string"},
        "Verb_Key_Words": {"type": "string"},
    },
    "required": ["Noun_Key_Words", "Verb_Key_Words"],
    "additionalProperties": False,
}


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def normalize_lyric_lines(full_lyrics: str) -> list[str]:
    """
    Convert raw lyrics (often LRC format) into a list of non-empty lyric lines.
    - Strips leading LRC timestamps like "[00:12.345]".
    - Drops empty lines.
    """
    lines: list[str] = []
    for raw in full_lyrics.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = _LRC_TIMESTAMP_RE.sub("", s).strip()
        if not s:
            continue
        lines.append(s)
    return lines


def _extract_converse_text(response: dict) -> str:
    text_response = ""
    content_list = response.get("output", {}).get(
        "message", {}).get("content", []) or []
    for content in content_list:
        if "text" in content:
            text_response += content["text"]
    return text_response


class Nova2LiteModel:

    def __init__(self):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        )
        # Cost tracking: Nova 2 Lite pricing
        # Input: $0.0003 per 1k tokens, Output: $0.0025 per 1k tokens
        self.costs = {
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    def _track_costs_from_usage(self, usage: dict) -> None:
        # This function intentionally does not `await`, so asyncio tasks won't yield mid-update.
        input_tokens = int(usage.get("inputTokens", 0) or 0)
        output_tokens = int(usage.get("outputTokens", 0) or 0)
        self.costs["input_tokens"] += input_tokens
        self.costs["output_tokens"] += output_tokens
        self.costs["input_cost"] += (input_tokens / 1000) * 0.0003
        self.costs["output_cost"] += (output_tokens / 1000) * 0.0025
        self.costs["total_cost"] = self.costs["input_cost"] + \
            self.costs["output_cost"]

    async def invoke_model(self, messages, system=None, max_tokens: int = 4096) -> str:
        system_prompts = [{"text": system}] if isinstance(
            system, str) else system

        response = await asyncio.to_thread(
            self.client.converse,
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=messages,
            system=system_prompts,
            inferenceConfig={
                "maxTokens": max_tokens,
            },
            additionalModelRequestFields={
                "reasoningConfig": {
                    "type": "disabled",  # enabled, disabled (default)
                }
            },
        )

        # Track costs
        self._track_costs_from_usage(response.get("usage", {}) or {})
        return _extract_converse_text(response)

    async def generate_vibe_card(self, media_path: str) -> str:
        media_bytes = read_bytes(media_path)
        media_format = media_path.split(".")[-1]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": media_format,
                            "source": {"bytes": media_bytes},
                        }
                    }
                ],
            }
        ]
        system_prompt = [{"text": VIBE_CARD_SYSTEM_PROMPT}]

        response = await asyncio.to_thread(
            self.client.converse,
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=messages,
            system=system_prompt,
            inferenceConfig={
                "maxTokens": 8192,
            },
            additionalModelRequestFields={
                "reasoningConfig": {
                    "type": "enabled",  # enabled, disabled (default)
                    "maxReasoningEffort": "low",
                }
            },
        )

        # Track costs
        self._track_costs_from_usage(response.get("usage", {}) or {})
        return _extract_converse_text(response)

    async def generate_recall_card(self, media_path: str) -> str:
        def _parse_recall_card_from_text(raw: str) -> Dict[str, str]:
            """
            Fallback parser for recall card outputs formatted like:
            <tools>
            <__function=recall_card_extractor>
                <__parameter=Noun_Key_Words>cat, dog</__parameter>
                <__parameter=Verb_Key_Words>run, jump</__parameter>
            </__function>
            Handles minor formatting issues and returns a dict.
            """
            params: Dict[str, str] = {}
            # First try well-formed parameter blocks.
            for key, val in re.findall(
                r"<__parameter=([A-Za-z0-9_]+)>(.*?)</__parameter>", raw, flags=re.DOTALL
            ):
                params[key.strip()] = val.strip()
            if params:
                return params

            # Fallback: line-based parsing to handle missing closing tags.
            for line in raw.splitlines():
                if "<__parameter=" not in line:
                    continue
                try:
                    _, rest = line.split("<__parameter=", 1)
                    key, remainder = rest.split(">", 1)
                    key = key.strip()
                    value = remainder.split("</__parameter", 1)[0].strip()
                    if key and value:
                        params[key] = value
                except ValueError:
                    continue
            return params

        media_bytes = read_bytes(media_path)
        media_format = media_path.split(".")[-1]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": media_format,
                            "source": {"bytes": media_bytes},
                        }
                    }
                ],
            },
        ]
        system_prompt = [{"text": RECALL_CARD_SYSTEM_PROMPT}]

        response = await asyncio.to_thread(
            self.client.converse,
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=messages,
            system=system_prompt,
            additionalModelRequestFields={
                "reasoningConfig": {
                    "type": "enabled",
                    "maxReasoningEffort": "low",
                }
            },
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": _RECALL_CARD_TOOL_NAME,
                            "description": "Extract noun and verb keywords from a video clip.",
                            "inputSchema": {
                                "json": _RECALL_CARD_JSON_SCHEMA,
                            },
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": _RECALL_CARD_TOOL_NAME}},
            },
        )

        # Track costs
        self._track_costs_from_usage(response.get("usage", {}) or {})
        return _parse_recall_card_from_text(_extract_converse_text(response))

    async def generate_song_augmented_queries_async(
        self,
        full_lyrics: str,
        use_prompt_cache_point: bool = True,
    ) -> list[str]:
        """
        Generate exactly one augmented query per lyric line by making one model call per line.
        Calls are run concurrently via asyncio, using a shared prompt prefix to enable prompt caching.
        """
        lyric_lines = normalize_lyric_lines(full_lyrics)
        return await self.generate_augmented_queries_for_lines(
            lyric_lines, use_prompt_cache_point=use_prompt_cache_point
        )

    async def generate_augmented_queries_for_lines(
        self,
        lyric_lines: list[str],
        use_prompt_cache_point: bool = True,
    ) -> list[str]:
        """
        Generate exactly one augmented query per lyric line using the provided list
        (preserves ordering without normalizing LRC timestamps).
        """
        system_prompt = [{"text": SONG_AUGMENTED_QUERY_SYSTEM_PROMPT}]
        cleaned_lines = [line.strip() for line in lyric_lines]
        if not cleaned_lines:
            return []

        numbered_lines = "\n".join(
            f"{i+1}. {line}" for i, line in enumerate(cleaned_lines)
        )
        shared_prefix_text = (
            "Full song lyrics (numbered lines):\n"
            f"{numbered_lines}\n\n"
            "You are generating an augmented visual retrieval query for a music video.\n"
            "The augmented query must be a dense, retrieval-friendly visual description matching a Vibe Card.\n"
        )

        def _build_messages(target_index: int, with_cache_point: bool) -> list[dict[str, Any]]:
            prefix_block: dict[str, Any] = {"text": shared_prefix_text}
            if with_cache_point:
                prefix_block["cachePoint"] = {"type": "default"}
            suffix_text = (
                f"Target line index: {target_index + 1}\n"
                f"Target lyric line: {cleaned_lines[target_index]}\n\n"
                "Return ONLY the augmented query for this target line as plain text.\n"
                "Do not include any labels, numbering, quotes, JSON, or additional commentary."
            )
            return [
                {"role": "user", "content": [prefix_block]},
                {"role": "user", "content": [{"text": suffix_text}]},
            ]

        async def _generate_one(idx: int) -> str:
            messages = _build_messages(
                idx, with_cache_point=use_prompt_cache_point)
            try:
                response = await asyncio.to_thread(
                    self.client.converse,
                    modelId="us.amazon.nova-2-lite-v1:0",
                    messages=messages,
                    system=system_prompt,
                    additionalModelRequestFields={
                        "reasoningConfig": {
                            "type": "enabled",
                            "maxReasoningEffort": "medium",
                        }
                    },
                )
            except (ParamValidationError, ClientError):
                if not use_prompt_cache_point:
                    raise
                messages = _build_messages(idx, with_cache_point=False)
                response = await asyncio.to_thread(
                    self.client.converse,
                    modelId="us.amazon.nova-2-lite-v1:0",
                    messages=messages,
                    system=system_prompt,
                    additionalModelRequestFields={
                        "reasoningConfig": {
                            "type": "enabled",
                            "maxReasoningEffort": "medium",
                        }
                    },
                )

            self._track_costs_from_usage(response.get("usage", {}) or {})
            return _extract_converse_text(response).strip()

        tasks = [_generate_one(i) for i in range(len(cleaned_lines))]

        queries = await tqdm.gather(
            *tasks,
            desc="Generating augmented queries",
            total=len(cleaned_lines),
        )

        if len(queries) != len(cleaned_lines):
            print(
                f"Warning: Generated {len(queries)} queries, expected {len(cleaned_lines)}"
            )

        return list(queries)


async def debug():
    model = Nova2LiteModel()

    full_lyrics = """
Lately I've been I've been losing sleep
Dreaming about the things that we could be
    """

    augmented_queries = await model.generate_song_augmented_queries_async(full_lyrics)
    import json
    print(json.dumps(augmented_queries, indent=2))

    print(f"\nCosts: {model.costs}")


if __name__ == "__main__":
    asyncio.run(debug())
