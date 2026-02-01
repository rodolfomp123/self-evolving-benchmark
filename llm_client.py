# -------------------------
# OpenAI-compatible LLM client
# -------------------------

import json
import requests
import time
from typing import Dict, List

# Constants
HTTP_ERROR_THRESHOLD = 400


def chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 700,
    timeout_s: int = 60,
    retries: int = 3,
    retry_backoff_s: float = 1.5,
) -> str:
    """
    Calls OpenAI-compatible POST /v1/chat/completions.

    Args:
        base_url: Base URL of the OpenAI-compatible API endpoint.
        api_key: API key for authentication.
        model: Model identifier to use for the completion.
        messages: List of message dicts with 'role' and 'content' keys.
        temperature: Sampling temperature (0.0-2.0). Defaults to 0.7.
        max_tokens: Maximum tokens in the response. Defaults to 700.
        timeout_s: Request timeout in seconds. Defaults to 60.
        retries: Number of retry attempts on failure. Defaults to 3.
        retry_backoff_s: Base backoff time in seconds between retries. Defaults to 1.5.

    Returns:
        The content string from the assistant's response message.

    Raises:
        RuntimeError: If all retry attempts fail due to HTTP errors,
            network issues, or malformed responses.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code >= HTTP_ERROR_THRESHOLD:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, json.JSONDecodeError, KeyError, RuntimeError) as e:
            if attempt < retries:
                time.sleep(retry_backoff_s * attempt)
            else:
                raise RuntimeError(f"chat_completions failed after {retries} attempts: {e}") from e
    
    raise RuntimeError("chat_completions: no retries configured")
