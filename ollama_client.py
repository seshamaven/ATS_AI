"""
Ollama client for chat completions and embeddings.
Provides OpenAI-compatible interface for Ollama API.
"""

import requests
import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama API.
    Provides OpenAI-compatible interface for chat completions.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:4b"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            model: Model name to use (default: qwen3:4b)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api"
        
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate chat completion using Ollama API.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (optional)
        
        Returns:
            Response dict with 'choices' containing message content
        """
        # Use Ollama's chat endpoint which properly handles message history
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            
            # Convert Ollama response to OpenAI-compatible format
            return {
                "choices": [{
                    "message": {
                        "role": result.get("message", {}).get("role", "assistant"),
                        "content": result.get("message", {}).get("content", "").strip()
                    }
                }]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
    
    def embeddings(self, input_text: str) -> List[float]:
        """
        Generate embeddings using Ollama API.
        
        Note: Not all models support embeddings. If the model doesn't support
        embeddings, this will raise an error.
        
        Args:
            input_text: Text to generate embeddings for
        
        Returns:
            List of embedding values
        """
        payload = {
            "model": self.model,
            "prompt": input_text
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama embeddings API request failed: {e}")
            # If embeddings are not supported, fall back to using sentence transformers
            logger.warning("Falling back to sentence transformers for embeddings")
            return self._fallback_embeddings(input_text)
    
    def _fallback_embeddings(self, text: str) -> List[float]:
        """
        Fallback to sentence transformers if Ollama embeddings are not available.
        
        Args:
            text: Text to generate embeddings for
        
        Returns:
            List of embedding values
        """
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight model for embeddings
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text, convert_to_numpy=True).tolist()
            # Pad or truncate to match expected dimension (1536)
            if len(embedding) < 1536:
                embedding.extend([0.0] * (1536 - len(embedding)))
            elif len(embedding) > 1536:
                embedding = embedding[:1536]
            return embedding
        except ImportError:
            logger.error("sentence-transformers not installed. Cannot generate embeddings.")
            raise RuntimeError("Embeddings not available and sentence-transformers not installed")


class OllamaChatCompletions:
    """
    OpenAI-compatible chat completions interface for Ollama.
    Supports both chat.create() and chat.completions.create() for compatibility.
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
        # Add self-reference for OpenAI-compatible interface: chat.completions.create()
        self.completions = self
    
    def create(self, model: str = None, messages: List[Dict[str, str]] = None, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Create chat completion (OpenAI-compatible interface).
        
        Args:
            model: Model name (ignored, uses client's model)
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            Response dict with 'choices' containing message content
        """
        return self.client.chat(messages, temperature, max_tokens)


class OllamaOpenAIClient:
    """
    OpenAI-compatible client wrapper for Ollama.
    Provides the same interface as OpenAI client with chat.completions.create()
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.chat = OllamaChatCompletions(ollama_client)


class OllamaEmbeddings:
    """
    OpenAI-compatible embeddings interface for Ollama.
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
    
    def create(self, input: str, model: str) -> Dict[str, Any]:
        """
        Create embeddings (OpenAI-compatible interface).
        
        Args:
            input: Input text
            model: Model name (ignored, uses client's model)
        
        Returns:
            Response dict with 'data' containing embedding
        """
        embedding = self.client.embeddings(input)
        return {
            "data": [{
                "embedding": embedding
            }]
        }


# Convenience function to get Ollama client
def get_ollama_client(base_url: Optional[str] = None, model: Optional[str] = None) -> OllamaClient:
    """
    Get Ollama client instance.
    
    Args:
        base_url: Optional base URL (defaults to http://localhost:11434)
        model: Optional model name (defaults to qwen3:4b)
    
    Returns:
        OllamaClient instance
    """
    import os
    base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    model = model or os.getenv('OLLAMA_MODEL', 'qwen3:4b')
    return OllamaClient(base_url=base_url, model=model)


def get_ollama_openai_client(base_url: Optional[str] = None, model: Optional[str] = None) -> OllamaOpenAIClient:
    """
    Get OpenAI-compatible Ollama client instance.
    
    Args:
        base_url: Optional base URL (defaults to http://localhost:11434)
        model: Optional model name (defaults to qwen3:4b)
    
    Returns:
        OllamaOpenAIClient instance with OpenAI-compatible interface
    """
    ollama_client = get_ollama_client(base_url=base_url, model=model)
    return OllamaOpenAIClient(ollama_client)

