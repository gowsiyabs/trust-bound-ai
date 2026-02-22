"""Device selection for embeddings"""

import torch
from ..utils.logging import get_logger

logger = get_logger(__name__)


def get_optimal_device() -> str:
    """Determine best device for embeddings"""
    
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS")
    else:
        device = "cpu"
        logger.info("Using CPU")
    
    return device
