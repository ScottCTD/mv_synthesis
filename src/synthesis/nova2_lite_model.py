import asyncio

import boto3
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

from synthesis.nova2_lite_utils import (
    _extract_converse_text,
    _sample_video_frames,
    normalize_lyric_lines,
    read_bytes,
)


VIBE_CARD_SYSTEM_PROMPT = """
# Role
You are an expert cinematic annotator for an animation archive. Your task is to analyze a short video clip and generate a structured metadata card.

# Objective
Provide a strictly structured output based ONLY on visible visual evidence. Do not hallucinate characters, objects, or narratives that are not present in the specific frames provided.

# Instructions & Field Definitions
You must provide the following four fields:

1. Scene Description:
   - Write 1-2 sentences describing the EXACT actions and objects visible.
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
You are an expert Music Video Editor. Your job is to rewrite ONE target lyric line into a dense retrieval query that matches our clip “Vibe Cards”.

Vibe Cards contain four parts:
- Scene Description (what we see: characters, actions, setting, key props)
- Vibe (mood/energy/pacing adjectives)
- Key Words (tags: objects, actions, locations, emotions)
- Musical Aspects (rhythm/beat-fit/action-sync/instrument-if-visible, comedic timing if applicable)

INPUT:
1) Full song lyrics as numbered lines
2) Target line index to rewrite

OUTPUT (IMPORTANT):
- Output EXACTLY ONE line of text (not JSON, no bullet list).
- Use this format INSIDE the single line:
  "Scene: ... | Vibe: ... | Key Words: ... | Musical Aspects: ..."
- Keep it compact but dense: ~40–80 words total.
- Use comma-separated phrases inside each field.

GOAL:
Maximize match to existing Vibe Cards for retrieval. Do not write poetry; write searchable visual descriptors.

HOW TO WRITE A GOOD QUERY:
1) Translate the lyric meaning into visual content:
   - Extract the core intent (literal or metaphorical).
   - Convert metaphors into concrete visuals and actions.

2) Scene Description should be concrete and clip-like:
   - Include (a) 1 setting, (b) 1–2 characters or subjects (use generic terms like “cat”, “mouse”, “character”, “pair”), (c) 1–2 actions, and (d) 2–4 salient props/objects.
   - Shot words like “close-up” or “wide shot” are allowed if they help retrieval.

3) Vibe should be 3–6 adjectives:
   - e.g., playful, mischievous, frantic, tense, dreamy, triumphant, chaotic, sneaky, tender.

4) Key Words should be 8–14 short tags:
   - Include characters/subjects, setting, props, actions, emotions.

5) Musical Aspects should be practical and short:
   - Beat-synced actions (hits, jumps, door slam), staccato vs legato motion,
   - “comedic timing” / “Mickey Mousing” if it fits,
   - cut-point potential (good for a sharp cut / good for a smooth dissolve),
   - instrument visibility only if plausible.

ABOUT “ONE SHOT” VS “MULTIPLE SHOTS”:
- Default: describe ONE strong, specific shot that clearly expresses the lyric (best for precision).
- If the lyric is very abstract OR your first-shot idea depends on a rare object/setting, include a SECOND alternative shot after "OR:" inside the Scene field (only one alternative, not more).
  Example: "Scene: ... OR: ..."

SPECIAL CASES:
- If the target line is a sound token like <Knock_Sound> or <Flute_Like_Sound>, describe the likely visible cause, the mood, and how the action can sync to the sound.
- Ignore title/artist text; focus on the lyric meaning.

DOMAIN ALIGNMENT:
Assume clips are stylized cartoon scenes. Prefer generic subjects ("cat", "mouse", "character") and cartoon-friendly props/settings (door, couch, box, ball, window, lamp, desk, alley, river, bridge, sign, chain, candle, fire). Avoid modern-specific objects (smartphone, telescope, calculator) unless absolutely necessary.

OR USAGE:
Do not use "OR" by default. Use at most one OR only if the first scene relies on a rare prop/setting. If you use OR, the alternative must be equally concrete (actions + props), not a generic fallback.

MUSICAL ASPECTS:
Do not name specific instruments unless they are visible or explicitly implied by the sound token. Focus on edit-fit: beat-synced action, staccato hit, legato motion, comedic timing (Mickey Mousing), sharp cut vs smooth dissolve.

"""


class NonRetryableBedrockError(Exception):
    """Exception for non-retryable Bedrock errors - these should not be retried."""
    pass


def _is_retryable_error(exception: Exception) -> bool:
    """Check if a Bedrock exception should trigger a retry."""
    if isinstance(exception, ClientError):
        error_code = exception.response.get("Error", {}).get("Code", "")
        retryable_errors = [
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceUnavailableException",
            "RequestTimeoutException",
            "InternalServerError",
        ]
        return error_code in retryable_errors
    return False


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

    async def _converse_with_retry(self, *args, **kwargs):
        """Wrapper for client.converse with retry logic."""
        @retry(
            retry=retry_if_exception_type(ClientError),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call_converse():
            try:
                return self.client.converse(*args, **kwargs)
            except ClientError as e:
                # Convert non-retryable errors to a different exception type
                # so tenacity won't retry them
                if not _is_retryable_error(e):
                    raise NonRetryableBedrockError(str(e)) from e
                raise  # Re-raise retryable ClientErrors for tenacity to handle
        
        try:
            return await asyncio.to_thread(_call_converse)
        except NonRetryableBedrockError as e:
            # Re-raise as the original ClientError
            raise e.__cause__ if e.__cause__ else e

    async def invoke_model(self, messages, system=None, max_tokens: int = 4096) -> str:
        system_prompts = [{"text": system}] if isinstance(system, str) else system

        response = await self._converse_with_retry(
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

        response = await self._converse_with_retry(
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=messages,
            system=system_prompt,
            inferenceConfig={
                "maxTokens": 8192,
            },
            additionalModelRequestFields={
                "reasoningConfig": {
                    "type": "enabled",  # enabled, disabled (default)
                    "maxReasoningEffort": "medium",
                }
            },
        )

        # Track costs
        self._track_costs_from_usage(response.get("usage", {}) or {})
        return _extract_converse_text(response)

    async def generate_vibe_card_frames(
        self,
        video_path: str,
        fps: float | None = None,
        max_frames: int = 8,
        reasoning_effort: str = "medium",
    ) -> str:
        frame_bytes, timestamps = _sample_video_frames(
            video_path, max_frames=max_frames, fps=fps
        )
        frame_labels = [
            f"frame_{idx + 1:04d}.jpg@{timestamp}"
            for idx, timestamp in enumerate(timestamps)
        ]
        prompt_lines = ["These are sequential video frames in chronological order."]
        if frame_labels:
            prompt_lines.append("Times: " + ", ".join(frame_labels))
        prompt = "\n".join(prompt_lines)

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    *[
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": frame_bytes_item},
                            }
                        }
                        for frame_bytes_item in frame_bytes
                    ],
                ],
            }
        ]
        system_prompt = [{"text": VIBE_CARD_SYSTEM_PROMPT}]
        reasoning_config = {"type": "disabled"}
        if reasoning_effort and reasoning_effort != "disabled":
            reasoning_config = {
                "type": "enabled",
                "maxReasoningEffort": reasoning_effort,
            }

        response = await self._converse_with_retry(
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=messages,
            system=system_prompt,
            inferenceConfig={
                "maxTokens": 8192,
            },
            additionalModelRequestFields={
                "reasoningConfig": reasoning_config,
            },
        )

        # Track costs
        self._track_costs_from_usage(response.get("usage", {}) or {})
        return _extract_converse_text(response)

    async def analyze_video_frames(
        self,
        video_path: str,
        user_prompt: str,
        system_prompt: str,
        reasoning_effort,
        fps: float | None = None,
    ) -> str:
        frame_bytes, timestamps = _sample_video_frames(video_path, fps=fps)
        frame_labels = [
            f"frame_{idx + 1:04d}.jpg@{timestamp}"
            for idx, timestamp in enumerate(timestamps)
        ]
        prompt_lines = ["These are sequential video frames in chronological order."]
        if frame_labels:
            prompt_lines.append("Times: " + ", ".join(frame_labels))
        if user_prompt:
            prompt_lines.append(f"Task: {user_prompt}")
        prompt = "\n".join(prompt_lines)

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    *[
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": frame_bytes_item},
                            }
                        }
                        for frame_bytes_item in frame_bytes
                    ],
                ],
            }
        ]

        system_prompts = [{"text": system_prompt}] if system_prompt else None
        reasoning_config = {"type": "disabled"}
        if reasoning_effort and reasoning_effort != "disabled":
            reasoning_config = {
                "type": "enabled",
                "maxReasoningEffort": reasoning_effort,
            }

        response = await self._converse_with_retry(
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=messages,
            system=system_prompts,
            inferenceConfig={
                "maxTokens": 4096,
            },
            additionalModelRequestFields={
                "reasoningConfig": reasoning_config,
            },
        )

        self._track_costs_from_usage(response.get("usage", {}) or {})
        return _extract_converse_text(response)

    async def generate_augmented_queries(self, lyrics: str | list[str]) -> list[str]:
        """
        Generate exactly one augmented query per lyric line from raw lyrics or a list of lines.
        """
        lyric_lines = normalize_lyric_lines(lyrics) if isinstance(lyrics, str) else lyrics
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
        )

        tasks = [
            self._converse_with_retry(
                modelId="us.amazon.nova-2-lite-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": (
                                    f"{shared_prefix_text}"
                                    f"Target line index: {idx + 1}\n"
                                    f"Target lyric line: {line}\n\n"
                                    "Return ONLY the augmented query for this target line as plain text.\n"
                                    "Do not include any labels, numbering, quotes, JSON, or additional commentary."
                                )
                            }
                        ],
                    },
                ],
                system=system_prompt,
                additionalModelRequestFields={
                    "reasoningConfig": {
                        "type": "enabled",
                        "maxReasoningEffort": "medium",
                    }
                },
            )
            for idx, line in enumerate(cleaned_lines)
        ]
        if tqdm is None:
            responses = await asyncio.gather(*tasks)
        else:
            responses = await tqdm.gather(
                *tasks,
                desc="Generating augmented queries",
                total=len(cleaned_lines),
            )

        queries = []
        for response in responses:
            self._track_costs_from_usage(response.get("usage", {}) or {})
            queries.append(_extract_converse_text(response).strip())

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
    
    augmented_queries = await model.generate_augmented_queries(full_lyrics)
    import json
    print(json.dumps(augmented_queries, indent=2))
    
    print(f"\nCosts: {model.costs}")


if __name__ == "__main__":
    asyncio.run(debug())
