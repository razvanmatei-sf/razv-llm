# ComfyUI Razv LLM Node

Custom ComfyUI node for integrating Claude API with image and text capabilities.

## Features

- Text-based chat with Claude models
- Optional image input support
- Configurable model selection (Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku)
- Adjustable parameters (temperature, max tokens, system prompt)

## Installation

### Option 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "razv-llm"
3. Click Install

### Option 2: Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/razvanmatei-sf/razv-llm.git
cd razv-llm
pip install -r requirements.txt
```

## Configuration

Set your Anthropic API key using one of these methods:
1. Environment variable: `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`
2. Direct input in the node's API key field

## Usage

1. Add the "RazvLLMChat" node to your workflow
2. Connect an image (optional) and provide a text prompt
3. Configure model and parameters as needed
4. Execute to get Claude's response

## Node Parameters

- **prompt**: Text input for Claude
- **model**: Choose between Claude 3.5 Sonnet, Claude 3 Opus, or Claude 3 Haiku
- **temperature**: Control response randomness (0.0 - 1.0)
- **max_tokens**: Maximum response length
- **system_prompt**: Optional system message to guide Claude's behavior
- **api_key**: Your Anthropic API key (if not set via environment)

## Requirements

- ComfyUI
- Python 3.8+
- See `requirements.txt` for Python dependencies

## License

MIT