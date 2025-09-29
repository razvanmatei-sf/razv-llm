import os
import json
import requests
import io
import base64
from torch import Tensor
from typing import Optional
from PIL import Image
import numpy as np


# Latest Claude models (as of 2025)
claude_models = [
    "claude-opus-4-1-20250805",  # Claude Opus 4.1 - Most capable model
    "claude-opus-4-1",  # Alias for Opus 4.1
    "claude-sonnet-4-20250514",  # Claude Sonnet 4 - Latest Sonnet
    "claude-sonnet-4-0",  # Alias for Sonnet 4
    "claude-3-5-sonnet-20241022",  # Sonnet 3.5
    "claude-3-5-haiku-20241022",  # Haiku 3.5
]

# Google Gemini models (as of 2025)
gemini_models = [
    "gemini-2.5-pro",  # Most capable reasoning model
    "gemini-2.5-flash",  # Best price-performance
    "gemini-2.5-flash-lite",  # Fastest, most cost-efficient
]

# All available models
all_models = claude_models + gemini_models


# Utility functions
def numpy2pil(numpy_image: np.ndarray, mode=None) -> Image.Image:
    if numpy_image.dtype == np.float32 or numpy_image.dtype == np.float64:
        numpy_image = np.clip(numpy_image, 0, 1)
        numpy_image = (numpy_image * 255).astype(np.uint8)
    elif numpy_image.dtype != np.uint8:
        numpy_image = numpy_image.astype(np.uint8)
    
    if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 1:
        numpy_image = numpy_image.squeeze(axis=2)
    
    if mode:
        return Image.fromarray(numpy_image, mode=mode)
    elif len(numpy_image.shape) == 2:
        return Image.fromarray(numpy_image, mode="L")
    elif len(numpy_image.shape) == 3:
        if numpy_image.shape[2] == 3:
            return Image.fromarray(numpy_image, mode="RGB")
        elif numpy_image.shape[2] == 4:
            return Image.fromarray(numpy_image, mode="RGBA")
    
    return Image.fromarray(numpy_image)


def tensor2pil(image: Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)


def pil2base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def call_claude_api(api_key: str, model: str, prompt: str, system_prompt: str, image: Optional[Tensor],
                    max_tokens: int, temperature: float, seed: int = -1, endpoint: str = "https://api.anthropic.com"):
    """Single function to call Claude API with text and optional image

    Note: Claude API does not natively support seed parameter for reproducible outputs.
    The seed parameter is included for UI consistency but does not affect Claude's output.
    """

    # Build message content
    content = []

    # Add image if provided
    if image is not None:
        if len(image.shape) == 4:  # Batch of images - just use first one
            pil_image = tensor2pil(image[0])
        else:
            pil_image = tensor2pil(image)

        image_base64 = pil2base64(pil_image)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_base64
            }
        })

    # Add text
    content.append({
        "type": "text",
        "text": prompt
    })

    # Build request
    url = f"{endpoint}/v1/messages"
    data = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": content
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Add system prompt if provided
    if system_prompt and system_prompt.strip():
        data["system"] = system_prompt

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    # Make request
    response = requests.post(url, json=data, headers=headers, timeout=60)

    # Handle errors
    if response.status_code != 200:
        error_msg = f"API request failed with status {response.status_code}"
        try:
            error_data = response.json()
            if error_data.get("error"):
                error_msg = error_data["error"].get("message", error_msg)
        except:
            error_msg = f"{error_msg}: {response.text}"
        raise Exception(error_msg)

    response_data = response.json()

    if response_data.get("error"):
        raise Exception(response_data.get("error").get("message", "Unknown error"))

    # Extract text response
    return response_data["content"][0]["text"]


def call_gemini_api(api_key: str, model: str, prompt: str, system_prompt: str, image: Optional[Tensor],
                    max_tokens: int, temperature: float, seed: int = -1):
    """Single function to call Gemini API with text and optional image

    Supports seed parameter for reproducible outputs when seed != -1.
    """

    # Build contents array
    contents = []

    # Add system instruction if provided
    system_instruction = None
    if system_prompt and system_prompt.strip():
        system_instruction = {"parts": [{"text": system_prompt}]}

    # Build content parts
    parts = []

    # Add image if provided
    if image is not None:
        if len(image.shape) == 4:  # Batch of images - just use first one
            pil_image = tensor2pil(image[0])
        else:
            pil_image = tensor2pil(image)

        image_base64 = pil2base64(pil_image)
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": image_base64
            }
        })

    # Add text
    parts.append({
        "text": prompt
    })

    contents.append({
        "parts": parts
    })

    # Build request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    generation_config = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }

    # Add seed if specified (not -1)
    if seed != -1:
        generation_config["seed"] = seed

    data = {
        "contents": contents,
        "generationConfig": generation_config
    }

    # Add system instruction if provided
    if system_instruction:
        data["systemInstruction"] = system_instruction

    headers = {
        "x-goog-api-key": api_key,
        "content-type": "application/json"
    }

    # Make request
    response = requests.post(url, json=data, headers=headers, timeout=60)

    # Handle errors
    if response.status_code != 200:
        error_msg = f"Gemini API request failed with status {response.status_code}"
        try:
            error_data = response.json()
            if error_data.get("error"):
                error_msg = error_data["error"].get("message", error_msg)
        except:
            error_msg = f"{error_msg}: {response.text}"
        raise Exception(error_msg)

    response_data = response.json()

    if response_data.get("error"):
        raise Exception(response_data.get("error").get("message", "Unknown error"))

    # Extract text response from Gemini format
    if "candidates" in response_data and len(response_data["candidates"]) > 0:
        candidate = response_data["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
            return candidate["content"]["parts"][0]["text"]

    raise Exception("No valid response received from Gemini API")


# ComfyUI Node Class
class RazvLLMChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "model": (all_models, {"default": "claude-opus-4-1-20250805"}),
                "prompt": ("STRING", {"multiline": True}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 200000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible results. -1 for random seed. Note: Only works with Gemini models, Claude doesn't support seeds."
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "Razv LLM"

    def chat(self, api_key: str, model: str, prompt: str, max_tokens: int, temperature: float, seed: int,
             system_prompt: str = "", image: Optional[Tensor] = None):

        # Check for API key in environment if not provided
        if not api_key or api_key.strip() == "":
            if model in gemini_models:
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            else:
                api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")

        if not api_key:
            if model in gemini_models:
                raise Exception("Gemini API key is required. Provide it in the node or set GEMINI_API_KEY environment variable.")
            else:
                raise Exception("Claude API key is required. Provide it in the node or set ANTHROPIC_API_KEY environment variable.")

        # Route to appropriate API based on model
        if model in gemini_models:
            response = call_gemini_api(
                api_key=api_key.strip(),
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )
        else:
            response = call_claude_api(
                api_key=api_key.strip(),
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )

        return (response,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "RazvLLMChat": RazvLLMChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RazvLLMChat": "Razv LLM Chat (Claude & Gemini)",
}