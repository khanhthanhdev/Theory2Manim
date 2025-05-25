# OpenRouter Integration Setup Guide

## Overview
This setup enables the use of OpenRouter models in the Theory2Manim application via LiteLLM. OpenRouter provides access to multiple AI models through a single API.

## Setup Instructions

### 1. Get OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Get your API key from the dashboard

### 2. Environment Configuration
Add your OpenRouter API key to your `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OR_SITE_URL=https://theory2manim.com  # Optional: for usage tracking
OR_APP_NAME=Theory2Manim              # Optional: for usage tracking
```

### 3. Available OpenRouter Models
The following OpenRouter models are now available in Theory2Manim:

**OpenAI Models via OpenRouter:**
- `openrouter/openai/gpt-4o` - Latest GPT-4 model
- `openrouter/openai/gpt-4o-mini` - Faster, cost-effective option
- `openrouter/openai/gpt-3.5-turbo` - Reliable and fast

**Anthropic Models via OpenRouter:**
- `openrouter/anthropic/claude-3.5-sonnet` - Excellent reasoning
- `openrouter/anthropic/claude-3-haiku` - Quick responses

**Google Models via OpenRouter:**
- `openrouter/google/gemini-pro-1.5` - Google's advanced model

**Open Source Models via OpenRouter:**
- `openrouter/meta-llama/llama-3.1-70b-instruct` - Meta's flagship model
- `openrouter/deepseek/deepseek-r1:free` - Free reasoning model
- `openrouter/deepseek/deepseek-chat` - Advanced conversation
- `openrouter/qwen/qwen-2.5-72b-instruct` - Alibaba's model

### 4. Usage Examples

#### Command Line Usage:
```bash
python generate_video.py \
  --topic "Chain Rule" \
  --description "Explain the chain rule in calculus with examples" \
  --model "openrouter/openai/gpt-4o"
```

#### Gradio Interface:
1. Start the Gradio app: `python gradio_app.py`
2. Select an OpenRouter model from the dropdown
3. Enter your topic and description
4. Generate your video

#### Programmatic Usage:
```python
from mllm_tools.openrouter import OpenRouterWrapper

# Initialize wrapper
wrapper = OpenRouterWrapper(
    model_name="openrouter/openai/gpt-4o",
    temperature=0.7,
    print_cost=True
)

# Send messages
messages = [
    {"type": "text", "content": "Explain the derivative of x^2"}
]

response = wrapper(messages)
print(response)
```

### 5. Cost Tracking
OpenRouter models support cost tracking. Set `print_cost=True` to see usage costs:

```python
wrapper = OpenRouterWrapper(
    model_name="openrouter/openai/gpt-4o",
    print_cost=True  # Shows cost after each request
)
```

### 6. Testing the Integration
Run the test script to verify everything works:

```bash
python test_openrouter.py
```

### 7. Model Selection Guidelines

**For Educational Content:**
- `openrouter/openai/gpt-4o` - Best overall quality
- `openrouter/anthropic/claude-3.5-sonnet` - Excellent explanations

**For Cost-Effective Usage:**
- `openrouter/openai/gpt-4o-mini` - Good balance of quality/cost
- `openrouter/deepseek/deepseek-r1:free` - Free tier available

**For Specific Use Cases:**
- `openrouter/google/gemini-pro-1.5` - Good for visual content
- `openrouter/meta-llama/llama-3.1-70b-instruct` - Open source option

### 8. Troubleshooting

**Common Issues:**
1. **Invalid API Key**: Ensure `OPENROUTER_API_KEY` is set correctly
2. **Model Not Found**: Check model name format (must start with `openrouter/`)
3. **Rate Limiting**: OpenRouter has rate limits; wait and retry if needed
4. **Cost Concerns**: Monitor usage in OpenRouter dashboard

**Debug Mode:**
Enable verbose logging for troubleshooting:
```python
wrapper = OpenRouterWrapper(
    model_name="openrouter/openai/gpt-4o",
    verbose=True  # Enables debug logging
)
```

### 9. Integration Architecture

The OpenRouter integration follows the existing wrapper pattern:
- `OpenRouterWrapper` - Main wrapper class compatible with existing interface
- Uses LiteLLM under the hood for OpenRouter API calls
- Supports multimodal content (text, images)
- Integrates with Langfuse for logging (optional)
- Works with the existing ComponentFactory pattern

This ensures seamless integration with the existing Theory2Manim codebase while adding access to a wide variety of models through OpenRouter.
