#!/usr/bin/env python3
"""
This file contains the chat generation logic for the LEANN project,
supporting different backends like Ollama, Hugging Face Transformers, and a simulation mode.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    """Abstract base class for a generic Language Model (LLM) interface."""
    @abstractmethod
    def ask(self, prompt: str, **kwargs) -> str:
        """
        Sends a prompt to the LLM and returns the generated text.

        Args:
            prompt: The input prompt for the LLM.
            **kwargs: Additional keyword arguments for the LLM backend.

        Returns:
            The response string from the LLM.
        """
        pass

class OllamaChat(LLMInterface):
    """LLM interface for Ollama models."""
    def __init__(self, model: str = "llama3:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        logger.info(f"Initializing OllamaChat with model='{model}' and host='{host}'")
        try:
            import requests
            # Check if the Ollama server is responsive
            if host:
                requests.get(host)
        except ImportError:
            raise ImportError("The 'requests' library is required for Ollama. Please install it with 'pip install requests'.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {host}. Please ensure Ollama is running.")
            raise ConnectionError(f"Could not connect to Ollama at {host}. Please ensure Ollama is running.")

    def ask(self, prompt: str, **kwargs) -> str:
        import requests
        import json

        full_url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Keep it simple for now
            "options": kwargs
        }
        logger.info(f"Sending request to Ollama: {payload}")
        try:
            response = requests.post(full_url, data=json.dumps(payload))
            response.raise_for_status()
            
            # The response from Ollama can be a stream of JSON objects, handle this
            response_parts = response.text.strip().split('\n')
            full_response = ""
            for part in response_parts:
                if part:
                    json_part = json.loads(part)
                    full_response += json_part.get("response", "")
                    if json_part.get("done"):
                        break
            return full_response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return f"Error: Could not get a response from Ollama. Details: {e}"

class HFChat(LLMInterface):
    """LLM interface for local Hugging Face Transformers models."""
    def __init__(self, model_name: str = "deepseek-ai/deepseek-llm-7b-chat"):
        logger.info(f"Initializing HFChat with model='{model_name}'")
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("The 'transformers' library is required for Hugging Face models. Please install it with 'pip install transformers'.")
        
        self.pipeline = pipeline("text-generation", model=model_name)

    def ask(self, prompt: str, **kwargs) -> str:
        # Sensible defaults for text generation
        params = {
            "max_length": 500,
            "num_return_sequences": 1,
            **kwargs
        }
        logger.info(f"Generating text with Hugging Face model with params: {params}")
        results = self.pipeline(prompt, **params)
        
        # Handle different response formats from transformers
        if isinstance(results, list) and len(results) > 0:
            generated_text = results[0].get('generated_text', '') if isinstance(results[0], dict) else str(results[0])
        else:
            generated_text = str(results)
        
        # Extract only the newly generated portion by removing the original prompt
        if isinstance(generated_text, str) and generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            # Fallback: return the full response if prompt removal fails
            response = str(generated_text)
            
        return response

class OpenAIChat(LLMInterface):
    """LLM interface for OpenAI models."""
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        logger.info(f"Initializing OpenAI Chat with model='{model}'")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("The 'openai' library is required for OpenAI models. Please install it with 'pip install openai'.")

    def ask(self, prompt: str, **kwargs) -> str:
        # Default parameters for OpenAI
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]}
        }
        
        logger.info(f"Sending request to OpenAI with model {self.model}")
        
        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error communicating with OpenAI: {e}")
            return f"Error: Could not get a response from OpenAI. Details: {e}"

class SimulatedChat(LLMInterface):
    """A simple simulated chat for testing and development."""
    def ask(self, prompt: str, **kwargs) -> str:
        logger.info("Simulating LLM call...")
        print("Prompt sent to LLM (simulation):", prompt[:500] + "...")
        return "This is a simulated answer from the LLM based on the retrieved context."

def get_llm(llm_config: Optional[Dict[str, Any]] = None) -> LLMInterface:
    """
    Factory function to get an LLM interface based on configuration.

    Args:
        llm_config: A dictionary specifying the LLM type and its parameters.
                    Example: {"type": "ollama", "model": "llama3"}
                             {"type": "hf", "model": "distilgpt2"}
                             None (for simulation mode)

    Returns:
        An instance of an LLMInterface subclass.
    """
    if llm_config is None:
        logger.info("No LLM config provided, defaulting to simulated chat.")
        return SimulatedChat()

    llm_type = llm_config.get("type", "simulated")
    model = llm_config.get("model")
    
    logger.info(f"Attempting to create LLM of type='{llm_type}' with model='{model}'")

    if llm_type == "ollama":
        return OllamaChat(model=model or "llama3:8b", host=llm_config.get("host", "http://localhost:11434"))
    elif llm_type == "hf":
        return HFChat(model_name=model or "deepseek-ai/deepseek-llm-7b-chat")
    elif llm_type == "openai":
        return OpenAIChat(model=model or "gpt-4o", api_key=llm_config.get("api_key"))
    elif llm_type == "simulated":
        return SimulatedChat()
    else:
        raise ValueError(f"Unknown LLM type: '{llm_type}'")
