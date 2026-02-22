"""OpenAI LLM creation"""

from llama_index.llms.openai import OpenAI
from ..utils.config import LLMConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_openai_llm(config: LLMConfig):
    """Create OpenAI LLM instance"""
    logger.info(f"Creating OpenAI LLM: {config.model}")
    
    return OpenAI(
        model=config.model,
        temperature=config.temperature,
        timeout=config.timeout,
    )
