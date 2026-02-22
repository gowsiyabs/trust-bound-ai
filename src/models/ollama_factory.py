"""Ollama LLM creation"""

from llama_index.llms.ollama import Ollama
from ..utils.config import LLMConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_ollama_llm(config: LLMConfig):
    """Create Ollama LLM instance"""
    logger.info(f"Creating Ollama LLM: {config.model}")
    
    return Ollama(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        request_timeout=config.timeout,
    )
