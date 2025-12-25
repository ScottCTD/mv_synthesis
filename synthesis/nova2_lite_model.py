import boto3
import asyncio
import re
from typing import Any

from botocore.exceptions import ClientError, ParamValidationError


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
        queries = await asyncio.gather(*tasks)

        if len(queries) != len(cleaned_lines):
            print(
                f"Warning: Generated {len(queries)} queries, expected {len(cleaned_lines)}"
            )

        return list(queries)

async def debug():
    model = Nova2LiteModel()

    full_lyrics = """
[00:00.000]Counting Stars - OneRepublic
[00:00.190]Lyrics by：Ryan Tedder
[00:00.390]Composed by：Ryan Tedder
[00:00.590]Produced by：Ryan Tedder/Noel Zancanella
[00:00.790]Lately I've been I've been losing sleep
[00:05.499]Dreaming about the things that we could be
[00:09.390]But baby I've been I've been praying hard
[00:14.361]Said no more counting dollars
[00:16.362]We'll be counting stars
[00:19.286]Yeah we'll be counting stars
[00:38.059]I see this life like a swinging vine
[00:40.673]Swing my heart across the line
[00:42.585]In my face is flashing signs
[00:44.570]Seek it out and ye' shall find
[00:46.475]Old but I'm not that old
[00:48.511]Young but I'm not that bold
[00:50.434]And I don't think the world is sold
[00:52.540]On just doing what we're told
[00:55.039]I I I I feel something so right
[00:59.233]Doing the wrong thing
[01:02.680]I I I I feel something so wrong
[01:07.032]Doing the right thing
[01:10.459]I couldn't lie couldn't lie couldn't lie
[01:14.200]Everything that kills me makes me feel alive
[01:18.166]Lately I've been I've been losing sleep
[01:21.938]Dreaming about the things that we could be
[01:25.758]But baby I've been I've been praying hard
[01:29.766]Said no more counting dollars
[01:31.718]We'll be counting stars
[01:33.846]Lately I've been I've been losing sleep
[01:37.725]Dreaming about the things that we could be
[01:41.647]But baby I've been I've been praying hard
[01:45.567]Said no more counting dollars
[01:47.527]We'll be we'll be counting stars
[01:56.755]I feel your love and I feel it burn
[01:59.453]Down this river every turn
[02:01.338]Hope is our four-letter word
[02:03.222]Make that money watch it burn
[02:05.141]Old but I'm not that old
[02:07.349]Young but I'm not that bold
[02:09.134]And I don't think the world is sold
[02:11.282]On just doing what we're told
[02:13.441]I I I I feel something so wrong
[02:17.845]Doing the right thing
[02:21.283]I couldn't lie couldn't lie couldn't lie
[02:25.075]Everything that drowns me makes me wanna fly
[02:28.909]Lately I've been I've been losing sleep
[02:32.814]Dreaming about the things that we could be
[02:36.601]But baby I've been I've been praying hard
[02:40.599]Said no more counting dollars
[02:42.611]We'll be counting stars
[02:44.723]Lately I've been I've been losing sleep
[02:48.715]Dreaming about the things that we could be
[02:52.404]But baby I've been I've been praying hard
[02:56.339]Said no more counting dollars
[02:58.155]We'll be we'll be counting stars
[03:04.424]Take that money
[03:05.025]Watch it burn
[03:05.857]Sink in the river
[03:06.889]The lessons I've learned
[03:08.059]Take that money
[03:08.738]Watch it burn
[03:09.924]Sink in the river
[03:10.899]The lessons I've learned
[03:12.013]Take that money
[03:12.830]Watch it burn
[03:13.863]Sink in the river
[03:14.852]The lessons I've learned
[03:16.028]Take that money
[03:16.915]Watch it burn
[03:17.843]Sink in the river
[03:18.836]The lessons I've learned
[03:19.985]Everything that kills me
[03:25.865]Makes me feel alive
[03:27.066]Lately I've been I've been losing sleep
[03:30.767]Dreaming about the things that we could be
[03:34.629]But baby I've been I've been praying hard
[03:38.636]Said no more counting dollars
[03:40.541]We'll be counting stars
[03:42.485]Lately I've been I've been losing sleep
[03:46.661]Dreaming about the things that we could be
[03:50.510]But baby I've been I've been praying hard
[03:54.525]Said no more counting dollars
[03:56.466]We'll be we'll be counting stars
[03:59.072]Take that money
[03:59.624]Watch it burn
[04:00.112]Sink in the river
[04:00.816]The lessons I've learned
[04:02.233]Take that money
[04:03.034]Watch it burn
[04:03.971]Sink in the river
[04:05.002]The lessons I've learned
[04:06.185]Take that money
[04:06.889]Watch it burn
[04:07.681]Sink in the river
[04:08.993]The lessons I've learned
[04:09.831]Take that money
[04:10.886]Watch it burn
[04:11.663]Sink in the river
[04:12.601]The lessons I've learned
    """
    
    augmented_queries = await model.generate_song_augmented_queries_async(full_lyrics)
    import json
    print(json.dumps(augmented_queries, indent=2))
    
    print(f"\nCosts: {model.costs}")


if __name__ == "__main__":
    asyncio.run(debug())
