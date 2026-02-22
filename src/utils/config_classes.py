"""Safety and evaluation config models"""

from pydantic import BaseModel, Field

from .config_models import (
    EmbeddingConfig,
    ChunkingConfig,
    RetrievalConfig,
    LLMConfig,
    VectorStoreConfig,
)


class SafetyConfig(BaseModel):
    """Safety layer configuration"""
    enable_input_validation: bool = Field(default=True)
    enable_output_filtering: bool = Field(default=True)
    enable_pii_detection: bool = Field(default=True)
    enable_hallucination_detection: bool = Field(default=True)

    jailbreak_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    hallucination_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_query_length: int = Field(default=2000, ge=1)


class RAGConfig(BaseModel):
    """Complete RAG system configuration"""
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    evaluator_llm: LLMConfig = Field(default_factory=LLMConfig)
    golden_dataset_path: str = Field(default="data/eval_datasets/golden_qa.csv")
    results_dir: str = Field(default="results/evaluations")


class AppConfig(BaseModel):
    """Main application configuration"""
    rag: RAGConfig = Field(default_factory=RAGConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    documents_dir: str = Field(default="data/documents")
    results_dir: str = Field(default="results")
    log_level: str = Field(default="INFO")
