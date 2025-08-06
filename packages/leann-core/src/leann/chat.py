#!/usr/bin/env python3
"""
This file contains the chat generation logic for the LEANN project,
supporting different backends like Ollama, Hugging Face Transformers, and a simulation mode.
"""

import difflib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_ollama_models() -> list[str]:
    """Check available Ollama models and return a list"""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except Exception:
        return []


def check_ollama_model_exists_remotely(model_name: str) -> tuple[bool, list[str]]:
    """Check if a model exists in Ollama's remote library and return available tags

    Returns:
        (model_exists, available_tags): bool and list of matching tags
    """
    try:
        import re

        import requests

        # Split model name and tag
        if ":" in model_name:
            base_model, requested_tag = model_name.split(":", 1)
        else:
            base_model, requested_tag = model_name, None

        # First check if base model exists in library
        library_response = requests.get("https://ollama.com/library", timeout=8)
        if library_response.status_code != 200:
            return True, []  # Assume exists if can't check

        # Extract model names from library page
        models_in_library = re.findall(r'href="/library/([^"]+)"', library_response.text)

        if base_model not in models_in_library:
            return False, []  # Base model doesn't exist

        # If base model exists, get available tags
        tags_response = requests.get(f"https://ollama.com/library/{base_model}/tags", timeout=8)
        if tags_response.status_code != 200:
            return True, []  # Base model exists but can't get tags

        # Extract tags for this model - be more specific to avoid HTML artifacts
        tag_pattern = rf"{re.escape(base_model)}:[a-zA-Z0-9\.\-_]+"
        raw_tags = re.findall(tag_pattern, tags_response.text)

        # Clean up tags - remove HTML artifacts and duplicates
        available_tags = []
        seen = set()
        for tag in raw_tags:
            # Skip if it looks like HTML (contains < or >)
            if "<" in tag or ">" in tag:
                continue
            if tag not in seen:
                seen.add(tag)
                available_tags.append(tag)

        # Check if exact model exists
        if requested_tag is None:
            # User just requested base model, suggest tags
            return True, available_tags[:10]  # Return up to 10 tags
        else:
            exact_match = model_name in available_tags
            return exact_match, available_tags[:10]

    except Exception:
        pass

    # If scraping fails, assume model might exist (don't block user)
    return True, []


def search_ollama_models_fuzzy(query: str, available_models: list[str]) -> list[str]:
    """Use intelligent fuzzy search for Ollama models"""
    if not available_models:
        return []

    query_lower = query.lower()
    suggestions = []

    # 1. Exact matches first
    exact_matches = [m for m in available_models if query_lower == m.lower()]
    suggestions.extend(exact_matches)

    # 2. Starts with query
    starts_with = [
        m for m in available_models if m.lower().startswith(query_lower) and m not in suggestions
    ]
    suggestions.extend(starts_with)

    # 3. Contains query
    contains = [m for m in available_models if query_lower in m.lower() and m not in suggestions]
    suggestions.extend(contains)

    # 4. Base model name matching (remove version numbers)
    def get_base_name(model_name: str) -> str:
        """Extract base name without version (e.g., 'llama3:8b' -> 'llama3')"""
        return model_name.split(":")[0].split("-")[0]

    query_base = get_base_name(query_lower)
    base_matches = [
        m
        for m in available_models
        if get_base_name(m.lower()) == query_base and m not in suggestions
    ]
    suggestions.extend(base_matches)

    # 5. Family/variant matching
    model_families = {
        "llama": ["llama2", "llama3", "alpaca", "vicuna", "codellama"],
        "qwen": ["qwen", "qwen2", "qwen3"],
        "gemma": ["gemma", "gemma2"],
        "phi": ["phi", "phi2", "phi3"],
        "mistral": ["mistral", "mixtral", "openhermes"],
        "dolphin": ["dolphin", "openchat"],
        "deepseek": ["deepseek", "deepseek-coder"],
    }

    query_family = None
    for family, variants in model_families.items():
        if any(variant in query_lower for variant in variants):
            query_family = family
            break

    if query_family:
        family_variants = model_families[query_family]
        family_matches = [
            m
            for m in available_models
            if any(variant in m.lower() for variant in family_variants) and m not in suggestions
        ]
        suggestions.extend(family_matches)

    # 6. Use difflib for remaining fuzzy matches
    remaining_models = [m for m in available_models if m not in suggestions]
    difflib_matches = difflib.get_close_matches(query_lower, remaining_models, n=3, cutoff=0.4)
    suggestions.extend(difflib_matches)

    return suggestions[:8]  # Return top 8 suggestions


# Remove this function entirely - we don't need external API calls for Ollama


# Remove this too - no need for fallback


def suggest_similar_models(invalid_model: str, available_models: list[str]) -> list[str]:
    """Use difflib to find similar model names"""
    if not available_models:
        return []

    # Get close matches using fuzzy matching
    suggestions = difflib.get_close_matches(invalid_model, available_models, n=3, cutoff=0.3)
    return suggestions


def check_hf_model_exists(model_name: str) -> bool:
    """Quick check if HuggingFace model exists without downloading"""
    try:
        from huggingface_hub import model_info

        model_info(model_name)
        return True
    except Exception:
        return False


def get_popular_hf_models() -> list[str]:
    """Return a list of popular HuggingFace models for suggestions"""
    try:
        from huggingface_hub import list_models

        # Get popular text-generation models, sorted by downloads
        models = list_models(
            filter="text-generation",
            sort="downloads",
            direction=-1,
            limit=20,  # Get top 20 most downloaded
        )

        # Extract model names and filter for chat/conversation models
        model_names = []
        chat_keywords = ["chat", "instruct", "dialog", "conversation", "assistant"]

        for model in models:
            model_name = model.id if hasattr(model, "id") else str(model)
            # Prioritize models with chat-related keywords
            if any(keyword in model_name.lower() for keyword in chat_keywords):
                model_names.append(model_name)
            elif len(model_names) < 10:  # Fill up with other popular models
                model_names.append(model_name)

        return model_names[:10] if model_names else _get_fallback_hf_models()

    except Exception:
        # Fallback to static list if API call fails
        return _get_fallback_hf_models()


def _get_fallback_hf_models() -> list[str]:
    """Fallback list of popular HuggingFace models"""
    return [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "facebook/blenderbot-400M-distill",
        "microsoft/phi-2",
        "deepseek-ai/deepseek-llm-7b-chat",
        "microsoft/DialoGPT-small",
        "facebook/blenderbot_small-90M",
        "microsoft/phi-1_5",
        "facebook/opt-350m",
        "EleutherAI/gpt-neo-1.3B",
    ]


def search_hf_models_fuzzy(query: str, limit: int = 10) -> list[str]:
    """Use HuggingFace Hub's native fuzzy search for model suggestions"""
    try:
        from huggingface_hub import list_models

        # HF Hub's search is already fuzzy! It handles typos and partial matches
        models = list_models(
            search=query,
            filter="text-generation",
            sort="downloads",
            direction=-1,
            limit=limit,
        )

        model_names = [model.id if hasattr(model, "id") else str(model) for model in models]

        # If direct search doesn't return enough results, try some variations
        if len(model_names) < 3:
            # Try searching for partial matches or common variations
            variations = []

            # Extract base name (e.g., "gpt3" from "gpt-3.5")
            base_query = query.lower().replace("-", "").replace(".", "").replace("_", "")
            if base_query != query.lower():
                variations.append(base_query)

            # Try common model name patterns
            if "gpt" in query.lower():
                variations.extend(["gpt2", "gpt-neo", "gpt-j", "dialoGPT"])
            elif "llama" in query.lower():
                variations.extend(["llama2", "alpaca", "vicuna"])
            elif "bert" in query.lower():
                variations.extend(["roberta", "distilbert", "albert"])

            # Search with variations
            for var in variations[:2]:  # Limit to 2 variations to avoid too many API calls
                try:
                    var_models = list_models(
                        search=var,
                        filter="text-generation",
                        sort="downloads",
                        direction=-1,
                        limit=3,
                    )
                    var_names = [
                        model.id if hasattr(model, "id") else str(model) for model in var_models
                    ]
                    model_names.extend(var_names)
                except Exception:
                    continue

        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in model_names:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)

        return unique_models[:limit]

    except Exception:
        # If search fails, return empty list
        return []


def search_hf_models(query: str, limit: int = 10) -> list[str]:
    """Simple search for HuggingFace models based on query (kept for backward compatibility)"""
    return search_hf_models_fuzzy(query, limit)


def validate_model_and_suggest(model_name: str, llm_type: str) -> str | None:
    """Validate model name and provide suggestions if invalid"""
    if llm_type == "ollama":
        available_models = check_ollama_models()
        if available_models and model_name not in available_models:
            error_msg = f"Model '{model_name}' not found in your local Ollama installation."

            # Check if the model exists remotely and get available tags
            model_exists_remotely, available_tags = check_ollama_model_exists_remotely(model_name)

            if model_exists_remotely and model_name in available_tags:
                # Exact model exists remotely - suggest pulling it
                error_msg += "\n\nTo install the requested model:\n"
                error_msg += f"  ollama pull {model_name}\n"

                # Show local alternatives
                suggestions = search_ollama_models_fuzzy(model_name, available_models)
                if suggestions:
                    error_msg += "\nOr use one of these similar installed models:\n"
                    for i, suggestion in enumerate(suggestions, 1):
                        error_msg += f"  {i}. {suggestion}\n"

            elif model_exists_remotely and available_tags:
                # Base model exists but requested tag doesn't - suggest correct tags
                base_model = model_name.split(":")[0]
                requested_tag = model_name.split(":", 1)[1] if ":" in model_name else None

                error_msg += (
                    f"\n\nModel '{base_model}' exists, but tag '{requested_tag}' is not available."
                )
                error_msg += f"\n\nAvailable {base_model} models you can install:\n"
                for i, tag in enumerate(available_tags[:8], 1):
                    error_msg += f"  {i}. ollama pull {tag}\n"
                if len(available_tags) > 8:
                    error_msg += f"  ... and {len(available_tags) - 8} more variants\n"

                # Also show local alternatives
                suggestions = search_ollama_models_fuzzy(model_name, available_models)
                if suggestions:
                    error_msg += "\nOr use one of these similar installed models:\n"
                    for i, suggestion in enumerate(suggestions, 1):
                        error_msg += f"  {i}. {suggestion}\n"

            else:
                # Model doesn't exist remotely - show fuzzy suggestions
                suggestions = search_ollama_models_fuzzy(model_name, available_models)
                error_msg += f"\n\nModel '{model_name}' was not found in Ollama's library."

                if suggestions:
                    error_msg += (
                        "\n\nDid you mean one of these installed models?\n"
                        + "\nTry to use ollama pull to install the model you need\n"
                    )

                    for i, suggestion in enumerate(suggestions, 1):
                        error_msg += f"  {i}. {suggestion}\n"
                else:
                    error_msg += "\n\nYour installed models:\n"
                    for i, model in enumerate(available_models[:8], 1):
                        error_msg += f"  {i}. {model}\n"
                    if len(available_models) > 8:
                        error_msg += f"  ... and {len(available_models) - 8} more\n"

            error_msg += "\n\nCommands:"
            error_msg += "\n  ollama list                    # List installed models"
            if model_exists_remotely and available_tags:
                if model_name in available_tags:
                    error_msg += f"\n  ollama pull {model_name}          # Install requested model"
                else:
                    error_msg += (
                        f"\n  ollama pull {available_tags[0]}    # Install recommended variant"
                    )
            error_msg += "\n  https://ollama.com/library     # Browse available models"
            return error_msg

    elif llm_type == "hf":
        # For HF models, we can do a quick existence check
        if not check_hf_model_exists(model_name):
            # Use HF Hub's native fuzzy search directly
            search_suggestions = search_hf_models_fuzzy(model_name, limit=8)

            error_msg = f"Model '{model_name}' not found on HuggingFace Hub."
            if search_suggestions:
                error_msg += "\n\nDid you mean one of these?\n"
                for i, suggestion in enumerate(search_suggestions, 1):
                    error_msg += f"  {i}. {suggestion}\n"
            else:
                # Fallback to popular models if search returns nothing
                popular_models = get_popular_hf_models()
                error_msg += "\n\nPopular chat models:\n"
                for i, model in enumerate(popular_models[:5], 1):
                    error_msg += f"  {i}. {model}\n"

            error_msg += f"\nSearch more: https://huggingface.co/models?search={model_name}&pipeline_tag=text-generation"
            return error_msg

    return None  # Model is valid or we can't check


class LLMInterface(ABC):
    """Abstract base class for a generic Language Model (LLM) interface."""

    @abstractmethod
    def ask(self, prompt: str, **kwargs) -> str:
        """
        Additional keyword arguments (kwargs) for advanced search customization. Example usage:
            chat.ask(
                "What is ANN?",
                top_k=10,
                complexity=64,
                beam_width=8,
                USE_DEFERRED_FETCH=True,
                skip_search_reorder=True,
                recompute_beighbor_embeddings=True,
                dedup_node_dis=True,
                prune_ratio=0.1,
                batch_recompute=True,
                global_pruning=True
            )

        Supported kwargs:
            - complexity (int): Search complexity parameter (default: 32)
            - beam_width (int): Beam width for search (default: 4)
            - USE_DEFERRED_FETCH (bool): Enable deferred fetch mode (default: False)
            - skip_search_reorder (bool): Skip search reorder step (default: False)
            - recompute_beighbor_embeddings (bool): Enable ZMQ embedding server for neighbor recomputation (default: False)
            - dedup_node_dis (bool): Deduplicate nodes by distance (default: False)
            - prune_ratio (float): Pruning ratio for search (default: 0.0)
            - batch_recompute (bool): Enable batch recomputation (default: False)
            - global_pruning (bool): Enable global pruning (default: False)
        """

        # """
        # Sends a prompt to the LLM and returns the generated text.

        # Args:
        #     prompt: The input prompt for the LLM.
        #     **kwargs: Additional keyword arguments for the LLM backend.

        # Returns:
        #     The response string from the LLM.
        # """
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

            # Pre-check model availability with helpful suggestions
            model_error = validate_model_and_suggest(model, "ollama")
            if model_error:
                raise ValueError(model_error)

        except ImportError:
            raise ImportError(
                "The 'requests' library is required for Ollama. Please install it with 'pip install requests'."
            )
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {host}. Please ensure Ollama is running.")
            raise ConnectionError(
                f"Could not connect to Ollama at {host}. Please ensure Ollama is running."
            )

    def ask(self, prompt: str, **kwargs) -> str:
        import json

        import requests

        full_url = f"{self.host}/api/generate"

        # Handle thinking budget for reasoning models
        options = kwargs.copy()
        thinking_budget = kwargs.get("thinking_budget")
        if thinking_budget:
            # Remove thinking_budget from options as it's not a standard Ollama option
            options.pop("thinking_budget", None)
            # Only apply reasoning parameters to models that support it
            reasoning_supported_models = [
                "gpt-oss:20b",
                "gpt-oss:120b",
                "deepseek-r1",
                "deepseek-coder",
            ]

            if thinking_budget in ["low", "medium", "high"]:
                if any(model in self.model.lower() for model in reasoning_supported_models):
                    options["reasoning"] = {"effort": thinking_budget, "exclude": False}
                    logger.info(f"Applied reasoning effort={thinking_budget} to model {self.model}")
                else:
                    logger.warning(
                        f"Thinking budget '{thinking_budget}' requested but model '{self.model}' may not support reasoning parameters. Proceeding without reasoning."
                    )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Keep it simple for now
            "options": options,
        }
        logger.debug(f"Sending request to Ollama: {payload}")
        try:
            logger.info("Sending request to Ollama and waiting for response...")
            response = requests.post(full_url, data=json.dumps(payload))
            response.raise_for_status()

            # The response from Ollama can be a stream of JSON objects, handle this
            response_parts = response.text.strip().split("\n")
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
    """LLM interface for local Hugging Face Transformers models with proper chat templates."""

    def __init__(self, model_name: str = "deepseek-ai/deepseek-llm-7b-chat"):
        logger.info(f"Initializing HFChat with model='{model_name}'")

        # Pre-check model availability with helpful suggestions
        model_error = validate_model_and_suggest(model_name, "hf")
        if model_error:
            raise ValueError(model_error)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The 'transformers' and 'torch' libraries are required for Hugging Face models. Please install them with 'pip install transformers torch'."
            )

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA is available. Using GPU.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("MPS is available. Using Apple Silicon GPU.")
        else:
            self.device = "cpu"
            logger.info("No GPU detected. Using CPU.")

        # Load tokenizer and model with timeout protection
        try:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Model download/loading timed out")

            # Set timeout for model loading (60 seconds)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                logger.info(f"Loading tokenizer for {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                logger.info(f"Loading model {model_name}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map="auto" if self.device != "cpu" else None,
                    trust_remote_code=True,
                )
                logger.info(f"Successfully loaded {model_name}")
            finally:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler

        except TimeoutError:
            logger.error(f"Model loading timed out for {model_name}")
            raise RuntimeError(
                f"Model loading timed out for {model_name}. Please check your internet connection or try a smaller model."
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

        # Move model to device if not using device_map
        if self.device != "cpu" and "device_map" not in str(self.model):
            self.model = self.model.to(self.device)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def ask(self, prompt: str, **kwargs) -> str:
        print("kwargs in HF: ", kwargs)
        # Check if this is a Qwen model and add /no_think by default
        is_qwen_model = "qwen" in self.model.config._name_or_path.lower()

        # For Qwen models, automatically add /no_think to the prompt
        if is_qwen_model and "/no_think" not in prompt and "/think" not in prompt:
            prompt = prompt + " /no_think"

        # Prepare chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using raw prompt: {e}")
                formatted_prompt = prompt
        else:
            # Fallback for models without chat template
            formatted_prompt = prompt

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Move inputs to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Set generation parameters
        generation_config = {
            "max_new_tokens": kwargs.get("max_tokens", kwargs.get("max_new_tokens", 512)),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("temperature", 0.7) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Handle temperature=0 for greedy decoding
        if generation_config["temperature"] == 0.0:
            generation_config["do_sample"] = False
            generation_config.pop("temperature")

        logger.info(f"Generating with HuggingFace model, config: {generation_config}")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)

        # Decode response
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


class OpenAIChat(LLMInterface):
    """LLM interface for OpenAI models."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        logger.info(f"Initializing OpenAI Chat with model='{model}'")

        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "The 'openai' library is required for OpenAI models. Please install it with 'pip install openai'."
            )

    def ask(self, prompt: str, **kwargs) -> str:
        # Default parameters for OpenAI
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
        }

        # Handle max_tokens vs max_completion_tokens based on model
        max_tokens = kwargs.get("max_tokens", 1000)
        if "o3" in self.model or "o4" in self.model or "o1" in self.model:
            # o-series models use max_completion_tokens
            params["max_completion_tokens"] = max_tokens
            params["temperature"] = 1.0
        else:
            # Other models use max_tokens
            params["max_tokens"] = max_tokens

        # Handle thinking budget for reasoning models
        thinking_budget = kwargs.get("thinking_budget")
        if thinking_budget and thinking_budget in ["low", "medium", "high"]:
            # Check if this is an o-series model (partial match for model names)
            o_series_models = ["o3", "o3-mini", "o4-mini", "o1", "o3-pro", "o3-deep-research"]
            if any(model in self.model for model in o_series_models):
                # Use the correct OpenAI reasoning parameter format
                params["reasoning_effort"] = thinking_budget
                logger.info(f"Applied reasoning_effort={thinking_budget} to model {self.model}")
            else:
                logger.warning(
                    f"Thinking budget '{thinking_budget}' requested but model '{self.model}' may not support reasoning parameters. Proceeding without reasoning."
                )

        # Add other kwargs (excluding thinking_budget as it's handled above)
        for k, v in kwargs.items():
            if k not in ["max_tokens", "temperature", "thinking_budget"]:
                params[k] = v

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


def get_llm(llm_config: dict[str, Any] | None = None) -> LLMInterface:
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
        llm_config = {
            "type": "openai",
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }

    llm_type = llm_config.get("type", "openai")
    model = llm_config.get("model")

    logger.info(f"Attempting to create LLM of type='{llm_type}' with model='{model}'")

    if llm_type == "ollama":
        return OllamaChat(
            model=model or "llama3:8b",
            host=llm_config.get("host", "http://localhost:11434"),
        )
    elif llm_type == "hf":
        return HFChat(model_name=model or "deepseek-ai/deepseek-llm-7b-chat")
    elif llm_type == "openai":
        return OpenAIChat(model=model or "gpt-4o", api_key=llm_config.get("api_key"))
    elif llm_type == "simulated":
        return SimulatedChat()
    else:
        raise ValueError(f"Unknown LLM type: '{llm_type}'")
