import boto3
import asyncio
import re
from typing import Any

from botocore.exceptions import ClientError, ParamValidationError
try:
    from tqdm.asyncio import tqdm
except ImportError:  # pragma: no cover - optional progress bar
    tqdm = None


_LRC_TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}\.\d{3}\]\s*")


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


VIBE_CARD_SYSTEM_PROMPT = """
# Role
You are an expert cinematic annotator for an animation archive. Your task is to analyze a short video clip and generate a structured metadata card.

# Objective
Provide a strictly structured output based ONLY on visible visual evidence. Do not hallucinate characters, objects, or narratives that are not present in the specific frames provided.

# Instructions & Field Definitions
You must provide the following four fields:

1. Scene Description:
   - Write 1-2 sentences describing the EXACT action and objects visible.
   - Be specific (e.g., use "double-barreled shotgun" instead of "crowbar" if visible).

2. Vibe:
   - Describe the abstract mood, energy, and emotional tone.
   - Use adjectives that capture the feeling (e.g., "suspenseful," "predatory," "manic," "playful").

3. Key Words:
   - A comma-separated list of high-value retrieval terms.
   - Include: Characters present (e.g., Tom), Objects (e.g., Barn, Shotgun), Actions (e.g., Hiding, Peeking), and Concepts (e.g., Ambush).

4. Musical Aspects (If Any):
   - Describe the *visual rhythm* or implied audio of the action.
   - Look for "Mickey Mousing" (action synchronized to beat), silence, sudden impacts, or rhythmic repetition (e.g., "staccato movement," "slow build-up," "quiet before the storm").

# Constraints
- **Visual Grounding:** DO NOT invent a story. If the clip is just a cat looking around, describe a cat looking around.
- **Tone:** Clinical and descriptive.
- **Format:** Use the exact headers below.

# Output Format
Scene Description: [Text]
Vibe: [Text]
Key Words: [Text]
Musical Aspects (If Any): [Text]
"""


SONG_AUGMENTED_QUERY_SYSTEM_PROMPT = """
You are an expert Music Video Director and Visual Prompt Engineer. Your goal is to translate song lyrics into rich, visual search queries to retrieve video clips from a vector database.

### THE CONTEXT
We have a database of video clips. Each clip is indexed by a "Vibe Card"—a structured description containing:
1. Scene Description (Concrete visuals, action, lighting)
2. Vibe (Abstract mood, energy, pacing)
3. Key Words (Specific objects, colors, tags)

### YOUR TASK
You will be provided with the full song lyrics (as numbered lines) and a specific target line index.
Your job is to generate exactly ONE augmented visual query for that target line.

### HOW TO CONSTRUCT THE AUGMENTED QUERY
The Augmented Query must be a dense, descriptive string that attempts to match the content of a Vibe Card. Do not merely repeat the lyrics. You must translate the *meaning* of the lyrics into *visuals*.

Apply the following logic:
1.  **Translate Metaphors to Visuals:**
    * *Lyric:* "My heart is on fire"
    * *Visual:* "Intense flames, red lighting, burning object, passion, fast-paced movement, warm color palette."
    * *Do NOT* just say "heart on fire" if a literal heart isn't the goal. Think about what the scene *looks* like.
2.  **Capture the Vibe/Energy:**
    * Infer the emotion (melancholy, hype, aggressive, serene) and include those adjectives.
    * Describe the pacing (slow motion, chaotic cuts, static shot).
3.  **Identify Physical Objects & Settings:**
    * Extract concrete nouns. If the lyrics are abstract, hallucinate fitting scenarios (e.g., for a sad song: "rainy window," "empty street," "lonely figure").
4.  **Cinematographic Terms:**
    * Use terms like "close-up," "wide shot," "bokeh," "neon lighting," "black and white" if they fit the mood.
5. Consider the full song. Make sure each augmented query is consistent with each other and maintain the flow and vibe of the whole song.
"""


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
    content_list = response.get("output", {}).get("message", {}).get("content", []) or []
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
        self.costs["total_cost"] = self.costs["input_cost"] + self.costs["output_cost"]

    async def invoke_model(self, messages, system=None, max_tokens: int = 4096) -> str:
        system_prompts = [{"text": system}] if isinstance(system, str) else system

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
            messages = _build_messages(idx, with_cache_point=use_prompt_cache_point)
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
        if tqdm is None:
            queries = await asyncio.gather(*tasks)
        else:
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
