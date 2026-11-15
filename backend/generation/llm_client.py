import json
import logging
from typing import Generator, Optional
import httpx
from config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for local LLM inference via Ollama with automatic fallback support.
    """

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        primary_model: str = settings.OLLAMA_MODEL,
        fallback_model: str = settings.OLLAMA_FALLBACK_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        timeout: float = settings.LLM_TIMEOUT
    ):
        self.base_url = base_url.rstrip("/")
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.timeout = timeout
        self.active_model = primary_model
        self._check_available_model()

    def _check_available_model(self):
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    installed = [m["name"] for m in res.json().get("models", [])]
                    # Check if primary or exact tag match exists
                    has_primary = any(self.primary_model in name for name in installed)
                    has_fallback = any(self.fallback_model in name for name in installed)
                    
                    if has_primary:
                        self.active_model = self.primary_model
                    elif has_fallback:
                        logger.warning(f"Primary model {self.primary_model} not found. Falling back to {self.fallback_model}")
                        self.active_model = self.fallback_model
                    elif installed:
                        self.active_model = installed[0]
                        logger.warning(f"Neither model found. Using installed model {self.active_model}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {self.base_url}: {e}")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        payload = {
            "model": self.active_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
            }
        }
        if system:
            payload["system"] = system

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                return response.json().get("response", "").strip()
        except Exception as primary_error:
            logger.error(f"Error calling {self.active_model}: {primary_error}")
            # Try fallback model if different
            if self.active_model != self.fallback_model:
                try:
                    logger.info(f"Attempting fallback to {self.fallback_model}...")
                    payload["model"] = self.fallback_model
                    with httpx.Client(timeout=self.timeout) as client:
                        response = client.post(f"{self.base_url}/api/generate", json=payload)
                        response.raise_for_status()
                        self.active_model = self.fallback_model
                        return response.json().get("response", "").strip()
                except Exception as fb_err:
                    raise RuntimeError(f"Ollama generation failed on fallback: {fb_err}") from fb_err
            raise RuntimeError(f"Ollama generation failed: {primary_error}") from primary_error

    def stream_generate(self, prompt: str, system: Optional[str] = None) -> Generator[str, None, None]:
        payload = {
            "model": self.active_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
            }
        }
        if system:
            payload["system"] = system

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            yield data.get("response", "")
                            if data.get("done", False):
                                break
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"\n[Error generating response: {e}]"
