"""
Quick start script for RAG system

Run this to test your setup:
    python quickstart.py
"""

from pathlib import Path
from src.utils.config import get_global_config
from src.models.llm_factory import create_llm
from src.models.rag_model import create_rag_model


def main():
    print("=" * 60)
    print("AI Trust Lab - RAG Quick Start")
    print("=" * 60)
    
    # Test 1: Configuration
    print("\n1. Testing configuration...")
    try:
        config = get_global_config()
        print(f"   ✓ Config loaded")
        print(f"   - LLM Provider: {config.rag.llm.provider}")
        print(f"   - LLM Model: {config.rag.llm.model}")
        print(f"   - Embedding Model: {config.rag.embeddings.model_name}")
    except Exception as e:
        print(f"   ✗ Configuration error: {e}")
        return
    
    # Test 2: LLM
    print("\n2. Testing LLM...")
    try:
        llm = create_llm()
        response = llm.complete("Respond with 'OK' in one word")
        print(f"   ✓ LLM working: {str(response).strip()[:50]}")
    except Exception as e:
        print(f"   ✗ LLM error: {e}")
        print("   Make sure Ollama is running or OpenAI API key is set")
        return
    
    # Test 3: Documents
    print("\n3. Checking documents...")
    docs_dir = Path("data/documents")
    doc_count = len(list(docs_dir.glob("*"))) if docs_dir.exists() else 0
    print(f"   Found {doc_count} document(s) in {docs_dir}")
    
    if doc_count == 0:
        print("   ⚠️  No documents found!")
        print("   Download SEC 10-K filings: python scripts/download_sec_filings.py")
        print("   Or add documents to data/documents/")
        print("\n   You can still test the system without documents,")
        print("   but queries will not work properly.")
        return
    
    # Test 4: RAG System
    print("\n4. Initializing RAG system...")
    try:
        print("   (This may take a few minutes on first run...)")
        model = create_rag_model(Path("data/documents"), force_reindex=False)
        stats = model.get_stats()
        print(f"   ✓ RAG system ready")
        print(f"   - Indexed documents: {stats['num_nodes']} chunks")
        print(f"   - Status: {stats['status']}")
    except Exception as e:
        print(f"   ✗ RAG initialization error: {e}")
        return
    
    # Test 5: Sample Query
    print("\n5. Testing sample query...")
    try:
        result = model.query("What is a deductible?")
        print(f"   ✓ Query successful")
        print(f"\n   Question: What is a deductible?")
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Confidence: {result['metadata']['confidence']:.1%}")
        print(f"   Sources: {result['metadata']['num_sources']}")
    except Exception as e:
        print(f"   ✗ Query error: {e}")
        return
    
    # Success
    print("\n" + "=" * 60)
    print("✅ Quick start complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the chat interface (fast - uses pre-built index):")
    print("   streamlit run apps/rag_chat/app.py")
    print("\n2. Build index first if needed:")
    print("   python scripts/build_index.py")
    print("\n3. View the evaluation dashboard:")
    print("   streamlit run apps/rag_chat/pages/eval_dashboard.py")
    print("\n4. Run Trust Lab modules: see README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
