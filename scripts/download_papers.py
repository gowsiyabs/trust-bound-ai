"""
ArXiv Paper Download Script

Automatically downloads papers for the RAG evaluation project.
Papers are downloaded to data/raw/ and excluded from git.

Usage:
    python scripts/download_papers.py
"""

import requests
from pathlib import Path
import time
from typing import List, Dict
import hashlib

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PAPER REGISTRY
# Papers are organized by topic with ArXiv IDs and metadata
# ============================================================================

PAPER_REGISTRY = {
    "foundational_rag": [
        {
            "arxiv_id": "2005.11401",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "filename": "rag_original_2020.pdf",
            "year": 2020,
            "notes": "Original RAG paper by Meta AI"
        },
        {
            "arxiv_id": "2310.11511",
            "title": "Self-RAG: Learning to Retrieve, Generate, and Critique",
            "filename": "self_rag_2023.pdf",
            "year": 2023,
            "notes": "Self-reflective RAG with retrieval decisions"
        },
        {
            "arxiv_id": "2401.15884",
            "title": "Corrective Retrieval Augmented Generation",
            "filename": "crag_2024.pdf",
            "year": 2024,
            "notes": "CRAG - evaluates and corrects retrieval"
        },
        {
            "arxiv_id": "2302.00083",
            "title": "In-Context Retrieval-Augmented Language Models",
            "filename": "in_context_rag_2023.pdf",
            "year": 2023,
            "notes": "REPLUG approach"
        },
        {
            "arxiv_id": "2305.06983",
            "title": "Active Retrieval Augmented Generation",
            "filename": "active_rag_2023.pdf",
            "year": 2023,
            "notes": "Adaptive retrieval frequency"
        },
    ],
    
    "advanced_architectures": [
        {
            "arxiv_id": "2404.16130",
            "title": "From Local to Global: A Graph RAG Approach",
            "filename": "graph_rag_2024.pdf",
            "year": 2024,
            "notes": "Microsoft's Graph RAG for narratives"
        },
        {
            "arxiv_id": "2301.12652",
            "title": "REPLUG: Retrieval-Augmented Black-Box Language Models",
            "filename": "replug_2023.pdf",
            "year": 2023,
            "notes": "Retrieval for black-box LLMs"
        },
        {
            "arxiv_id": "2402.16827",
            "title": "The Unreasonable Effectiveness of Retrieval-Augmented Generation",
            "filename": "rag_effectiveness_2024.pdf",
            "year": 2024,
            "notes": "Survey of RAG effectiveness"
        },
    ],
    
    "evaluation": [
        {
            "arxiv_id": "2309.15217",
            "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
            "filename": "ragas_2023.pdf",
            "year": 2023,
            "notes": "RAGAS evaluation framework"
        },
        {
            "arxiv_id": "2311.09476",
            "title": "ARES: An Automated Evaluation Framework for RAG Systems",
            "filename": "ares_2023.pdf",
            "year": 2023,
            "notes": "ARES with confidence scoring"
        },
        {
            "arxiv_id": "2306.05685",
            "title": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
            "filename": "llm_as_judge_2023.pdf",
            "year": 2023,
            "notes": "LLM-based evaluation methodology"
        },
        {
            "arxiv_id": "2303.16634",
            "title": "G-Eval: NLG Evaluation using GPT-4",
            "filename": "g_eval_2023.pdf",
            "year": 2023,
            "notes": "GPT-based evaluation with chain-of-thought"
        },
        {
            "arxiv_id": "2305.14251",
            "title": "FActScore: Fine-grained Atomic Evaluation of Factual Precision",
            "filename": "factscore_2023.pdf",
            "year": 2023,
            "notes": "Atomic fact-level evaluation"
        },
        {
            "arxiv_id": "2404.13781",
            "title": "Evaluating Retrieval Quality in Retrieval-Augmented Generation",
            "filename": "rag_retrieval_eval_2024.pdf",
            "year": 2024,
            "notes": "Focus on retrieval metrics"
        },
    ],
    
    "retrieval_methods": [
        {
            "arxiv_id": "2004.04906",
            "title": "Dense Passage Retrieval for Open-Domain Question Answering",
            "filename": "dpr_2020.pdf",
            "year": 2020,
            "notes": "DPR - foundational dense retrieval"
        },
        {
            "arxiv_id": "2212.10496",
            "title": "Precise Zero-Shot Dense Retrieval without Relevance Labels",
            "filename": "hyde_2022.pdf",
            "year": 2022,
            "notes": "HyDE - hypothetical document embeddings"
        },
        {
            "arxiv_id": "2307.03172",
            "title": "Lost in the Middle: How Language Models Use Long Contexts",
            "filename": "lost_in_middle_2023.pdf",
            "year": 2023,
            "notes": "Context positioning effects in RAG"
        },
        {
            "arxiv_id": "2312.06648",
            "title": "Proposition-level Retrieval for RAG",
            "filename": "proposition_retrieval_2023.pdf",
            "year": 2023,
            "notes": "Atomic proposition chunking"
        },
    ],
    
    "surveys_and_benchmarks": [
        {
            "arxiv_id": "2312.10997",
            "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
            "filename": "rag_survey_2023.pdf",
            "year": 2023,
            "notes": "Comprehensive RAG survey"
        },
        {
            "arxiv_id": "2307.03109",
            "title": "A Survey on Evaluation of Large Language Models",
            "filename": "llm_eval_survey_2023.pdf",
            "year": 2023,
            "notes": "General LLM evaluation survey"
        },
        {
            "arxiv_id": "2402.19473",
            "title": "Retrieval-Augmented Generation for AI-Generated Content: A Survey",
            "filename": "rag_aigc_survey_2024.pdf",
            "year": 2024,
            "notes": "RAG for content generation"
        },
        {
            "arxiv_id": "2104.08663",
            "title": "BEIR: A Heterogeneous Benchmark for Information Retrieval",
            "filename": "beir_benchmark_2021.pdf",
            "year": 2021,
            "notes": "Retrieval benchmark dataset"
        },
    ]
}

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_arxiv_paper(arxiv_id: str, output_path: Path, max_retries: int = 3) -> bool:
    """
    Download a paper from ArXiv
    
    Args:
        arxiv_id: ArXiv ID (e.g., "2005.11401")
        output_path: Where to save the PDF
        max_retries: Number of retry attempts
    
    Returns:
        True if successful, False otherwise
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    for attempt in range(max_retries):
        try:
            print(f"  Downloading from {url}... ", end="", flush=True)
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Write to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify it's a valid PDF
            if output_path.stat().st_size < 1000:
                print(f"❌ (file too small, might be invalid)")
                output_path.unlink()
                return False
            
            print(f"✓ ({output_path.stat().st_size / 1024:.1f} KB)")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  Error: {e}")
                return False
    
    return False

def verify_paper(paper_path: Path) -> Dict[str, any]:
    """Verify and get metadata about a downloaded paper"""
    if not paper_path.exists():
        return {"exists": False}
    
    size_kb = paper_path.stat().st_size / 1024
    
    # Basic PDF validation (check header)
    try:
        with open(paper_path, 'rb') as f:
            header = f.read(5)
            is_valid_pdf = header == b'%PDF-'
    except:
        is_valid_pdf = False
    
    return {
        "exists": True,
        "size_kb": size_kb,
        "is_valid_pdf": is_valid_pdf,
        "path": paper_path
    }

# ============================================================================
# MAIN DOWNLOAD LOGIC
# ============================================================================

def download_all_papers(force_redownload: bool = False, categories: List[str] = None):
    """
    Download all papers in the registry
    
    Args:
        force_redownload: Re-download even if file exists
        categories: List of categories to download (None = all)
    """
    print("="*70)
    print("RAG Evaluation Lab - Paper Download Script")
    print("="*70)
    print(f"Download directory: {DATA_DIR}")
    print()
    
    # Filter categories if specified
    if categories:
        papers_to_download = {
            cat: PAPER_REGISTRY[cat] 
            for cat in categories 
            if cat in PAPER_REGISTRY
        }
    else:
        papers_to_download = PAPER_REGISTRY
    
    total_papers = sum(len(papers) for papers in papers_to_download.values())
    downloaded = 0
    skipped = 0
    failed = 0
    
    for category, papers in papers_to_download.items():
        print(f"\n📁 Category: {category}")
        print("-" * 70)
        
        for paper in papers:
            output_path = DATA_DIR / paper["filename"]
            
            print(f"\n{paper['title'][:60]}...")
            print(f"  ArXiv: {paper['arxiv_id']} | Year: {paper['year']}")
            
            # Check if already exists
            verification = verify_paper(output_path)
            
            if verification["exists"] and not force_redownload:
                if verification["is_valid_pdf"]:
                    print(f"  ⏭️  Already exists ({verification['size_kb']:.1f} KB) - skipping")
                    skipped += 1
                    continue
                else:
                    print(f"  ⚠️  Exists but invalid - re-downloading")
                    output_path.unlink()
            
            # Download
            success = download_arxiv_paper(paper["arxiv_id"], output_path)
            
            if success:
                downloaded += 1
            else:
                failed += 1
            
            # Be nice to ArXiv servers
            time.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    print(f"Total papers in registry: {total_papers}")
    print(f"✅ Downloaded:            {downloaded}")
    print(f"⏭️  Skipped (existed):     {skipped}")
    print(f"❌ Failed:                {failed}")
    print(f"\nPapers saved to: {DATA_DIR}")
    
    if failed > 0:
        print("\n⚠️  Some downloads failed. You can:")
        print("   1. Run this script again (it will retry failed ones)")
        print("   2. Manually download from ArXiv and place in data/raw/")

def list_papers():
    """List all papers in the registry"""
    print("="*70)
    print("Papers in Registry")
    print("="*70)
    
    for category, papers in PAPER_REGISTRY.items():
        print(f"\n📁 {category.upper().replace('_', ' ')}")
        print("-" * 70)
        for i, paper in enumerate(papers, 1):
            status = "✓" if verify_paper(DATA_DIR / paper["filename"])["exists"] else "⭕"
            print(f"{status} {i}. {paper['title']}")
            print(f"   ArXiv: {paper['arxiv_id']} → {paper['filename']}")
            if paper.get("notes"):
                print(f"   Notes: {paper['notes']}")
    
    total = sum(len(papers) for papers in PAPER_REGISTRY.values())
    downloaded = sum(
        1 for papers in PAPER_REGISTRY.values() 
        for paper in papers 
        if verify_paper(DATA_DIR / paper["filename"])["exists"]
    )
    print(f"\n{downloaded}/{total} papers downloaded")

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ArXiv papers for RAG evaluation project")
    parser.add_argument(
        "command",
        choices=["download", "list", "verify"],
        help="Command to run"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--category",
        nargs="+",
        choices=list(PAPER_REGISTRY.keys()),
        help="Download only specific categories"
    )
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_all_papers(
            force_redownload=args.force,
            categories=args.category
        )
    elif args.command == "list":
        list_papers()
    elif args.command == "verify":
        list_papers()  # Same output for now