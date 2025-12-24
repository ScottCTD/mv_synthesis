import boto3


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

    def invoke_model(self, messages, system=None, max_tokens: int = 4096) -> str:
        system_prompts = [{"text": system}] if isinstance(system, str) else system

        response = self.client.converse(
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
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        self.costs["input_tokens"] += input_tokens
        self.costs["output_tokens"] += output_tokens
        self.costs["input_cost"] += (input_tokens / 1000) * 0.0003
        self.costs["output_cost"] += (output_tokens / 1000) * 0.0025
        self.costs["total_cost"] = self.costs["input_cost"] + self.costs["output_cost"]

        text_response = ""
        content_list = response["output"]["message"]["content"]
        for content in content_list:
            if "text" in content:
                text_response += content["text"]
        return text_response

    def generate_video_vibe_card(self, video_path: str) -> str:
        video_bytes = read_bytes(video_path)
        video_format = video_path.split(".")[-1]
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": {"format": video_format, "source": {"bytes": video_bytes}}}
                ],
            }
        ]
        system_prompt = [{"text": VIBE_CARD_SYSTEM_PROMPT}]
        
        response = self.client.converse(
            modelId="us.amazon.nova-2-lite-v1:0",
            messages=messages,
            system=system_prompt,
            inferenceConfig={
                "maxTokens": 8192,
            },
            additionalModelRequestFields={
                "reasoningConfig": {
                    "type": "enabled",  # enabled, disabled (default)
                    "maxReasoningEffort": "low"
                }
            },
        )

        # Track costs
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        self.costs["input_tokens"] += input_tokens
        self.costs["output_tokens"] += output_tokens
        self.costs["input_cost"] += (input_tokens / 1000) * 0.0003
        self.costs["output_cost"] += (output_tokens / 1000) * 0.0025
        self.costs["total_cost"] = self.costs["input_cost"] + self.costs["output_cost"]

        text_response = ""
        content_list = response["output"]["message"]["content"]
        for content in content_list:
            if "text" in content:
                text_response += content["text"]
        
        return text_response


def debug():
    model = Nova2LiteModel()
    
    video_path = "/Users/scottcui/projects/mv_synthesis/datasets/ds1/videos/0/E01-Little_Quacker-Scene-009.mkv"
    
    vibe_card = model.generate_video_vibe_card(video_path)
    print(vibe_card)
    print(f"\nCosts: {model.costs}")


if __name__ == "__main__":
    debug()