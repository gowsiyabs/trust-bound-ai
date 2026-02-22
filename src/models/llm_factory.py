"""LLM factory - modularized version"""

from .ollama_factory import create_ollama_llm
from .openai_factory import create_openai_llm
from ..utils.config import LLMConfig, get_global_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_llm(config: LLMConfig = None):
    """Create LLM based on config"""
    
    if config is None:
        config = get_global_config().rag.llm
    
    provider = config.provider.lower()
    
    if provider == "ollama":
        return create_ollama_llm(config)
    elif provider == "openai":
        return create_openai_llm(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_evaluator_llm():
    """Create LLM for evaluation tasks"""
    config = get_global_config().evaluation.evaluator_llm
    return create_llm(config)
