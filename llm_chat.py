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
                    max_tokens: int, temperature: float, endpoint: str = "https://api.anthropic.com"):
    """Single function to call Claude API with text and optional image"""
    
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


# ComfyUI Node Class
class RazvLLMChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "model": (claude_models, {"default": "claude-opus-4-1-20250805"}),
                "prompt": ("STRING", {"multiline": True}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 200000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant."}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "Razv LLM"

    def chat(self, api_key: str, model: str, prompt: str, max_tokens: int, temperature: float, 
             system_prompt: str = "You are a helpful AI assistant.", image: Optional[Tensor] = None):
        
        # Check for API key in environment if not provided
        if not api_key or api_key.strip() == "":
            api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        
        if not api_key:
            raise Exception("Claude API key is required. Provide it in the node or set ANTHROPIC_API_KEY environment variable.")
        
        # Call Claude API
        response = call_claude_api(
            api_key=api_key.strip(),
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            image=image,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return (response,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "RazvLLMChat": RazvLLMChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RazvLLMChat": "Razv LLM Chat (Image Optional)",
}