import asyncio
import base64
import json
from pathlib import Path
from typing import Literal

import boto3

try:
    from synthesis.ffmpeg_utils import get_video_duration
except ImportError:
    from ffmpeg_utils import get_video_duration


def encode_b64(file_path) -> str:
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


class Nova2OmniEmbeddings:

    def __init__(self):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        )
        # Cost tracking: Nova Multimodal Embeddings pricing
        # Text: $0.000135 per 1k tokens
        # Video: $0.0007 per second
        # Audio: $0.00014 per second
        self.costs = {
            "text_tokens": 0,
            "video_seconds": 0.0,
            "audio_seconds": 0.0,
            "text_cost": 0.0,
            "video_cost": 0.0,
            "audio_cost": 0.0,
            "total_cost": 0.0,
        }

    async def _invoke_model(self, request_body):
        """
        Returns a dictionary with the following structure:
        {
            "request_id": str,
            "embeddings": [
                {
                    "embedding_type": str,
                    "embedding": list[float],
                }
            ]
        }
        """
        response = await asyncio.to_thread(
            self.client.invoke_model,
            body=json.dumps(request_body, indent=2),
            modelId="amazon.nova-2-multimodal-embeddings-v1:0",
            accept="application/json",
            contentType="application/json",
        )
        request_id = response.get("ResponseMetadata").get("RequestId")
        response_body = json.loads(response.get("body").read())

        results = response_body["embeddings"]
        for result in results:
            result["embedding_type"] = result.pop("embeddingType")

        # Check if usage info is in response_body for text embeddings
        if "usage" in response_body:
            usage = response_body["usage"]
            if "inputTokens" in usage:
                text_tokens = usage.get("inputTokens", 0)
                self.costs["text_tokens"] += text_tokens
                self.costs["text_cost"] += (text_tokens / 1000) * 0.000135
                self._update_total_cost()

        return {
            "request_id": request_id,
            "embeddings": results,
        }

    def _update_total_cost(self):
        """Update the total cost."""
        self.costs["total_cost"] = (
            self.costs["text_cost"]
            + self.costs["video_cost"]
            + self.costs["audio_cost"]
        )

    async def embed_text(
        self,
        text: str,
        embedding_purpose: Literal[
            "GENERIC_INDEX",
            "TEXT_RETRIEVAL",
            "IMAGE_RETRIEVAL",
            "VIDEO_RETRIEVAL",
            "AUDIO_RETRIEVAL",
            "DOCUMENT_RETRIEVAL",
            "GENERIC_RETRIEVAL",
            "CLASSIFICATION",
            "CLUSTERING",
        ] = "GENERIC_INDEX",
        embedding_dimension: int = 3072,
        truncation_mode: Literal["START", "END", "NONE"] = "NONE",
    ):
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": embedding_purpose,
                "embeddingDimension": embedding_dimension,
                "text": {"truncationMode": truncation_mode, "value": text},
            },
        }
        return await self._invoke_model(request_body)

    async def embed_video(
        self,
        video_path: str,
        embedding_purpose: Literal[
            "GENERIC_INDEX",
            "TEXT_RETRIEVAL",
            "IMAGE_RETRIEVAL",
            "VIDEO_RETRIEVAL",
            "AUDIO_RETRIEVAL",
            "DOCUMENT_RETRIEVAL",
            "GENERIC_RETRIEVAL",
            "CLASSIFICATION",
            "CLUSTERING",
        ] = "GENERIC_INDEX",
        embedding_dimension: int = 3072,
        embedding_mode: Literal[
            "AUDIO_VIDEO_COMBINED", "AUDIO_VIDEO_SEPARATE"
        ] = "AUDIO_VIDEO_COMBINED",
    ):
        """
        Args:
            video_path: str
            embedding_mode: Literal["AUDIO_VIDEO_COMBINED", "AUDIO_VIDEO_SEPARATE"]
                - "AUDIO_VIDEO_COMBINED" - Will produce a single embedding combining both audible and visual content.
                - "AUDIO_VIDEO_SEPARATE" - Will produce two embeddings, one for the audible content and one for the visual content.
        """
        video_path_obj = Path(video_path)
        video_foramt = video_path_obj.suffix[1:]
        video_b64 = encode_b64(video_path_obj)

        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": embedding_purpose,
                "embeddingDimension": embedding_dimension,
                "video": {
                    "format": video_foramt,
                    "embeddingMode": embedding_mode,
                    "source": {"bytes": video_b64},
                },
            },
        }
        result = await self._invoke_model(request_body)

        # Calculate video duration for cost tracking
        video_duration = get_video_duration(video_path_obj)
        if video_duration is not None:
            # For AUDIO_VIDEO_SEPARATE mode, we charge for both video and audio
            if embedding_mode == "AUDIO_VIDEO_SEPARATE":
                self.costs["video_seconds"] += video_duration
                self.costs["audio_seconds"] += video_duration
                self.costs["video_cost"] += video_duration * 0.0007
                self.costs["audio_cost"] += video_duration * 0.00014
            else:
                # For combined mode, charge for video (which includes audio)
                self.costs["video_seconds"] += video_duration
                self.costs["video_cost"] += video_duration * 0.0007
            self._update_total_cost()

        return result

    async def embed_audio(
        self,
        audio_path: str,
        embedding_purpose: Literal[
            "GENERIC_INDEX",
            "TEXT_RETRIEVAL",
            "IMAGE_RETRIEVAL",
            "VIDEO_RETRIEVAL",
            "AUDIO_RETRIEVAL",
            "DOCUMENT_RETRIEVAL",
            "GENERIC_RETRIEVAL",
            "CLASSIFICATION",
            "CLUSTERING",
        ] = "GENERIC_INDEX",
        embedding_dimension: int = 3072,
    ):
        audio_path_obj = Path(audio_path)
        audio_format = audio_path_obj.suffix[1:]
        audio_b64 = encode_b64(audio_path_obj)

        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": embedding_purpose,
                "embeddingDimension": embedding_dimension,
                "audio": {"format": audio_format, "source": {"bytes": audio_b64}},
            },
        }
        result = await self._invoke_model(request_body)

        # Calculate audio duration for cost tracking
        audio_duration = get_video_duration(
            audio_path_obj
        )  # ffprobe works for audio too
        if audio_duration is not None:
            self.costs["audio_seconds"] += audio_duration
            self.costs["audio_cost"] += audio_duration * 0.00014
            self._update_total_cost()

        return result
