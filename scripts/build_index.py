"""
Build RAG vector index from documents. Run this once (or when documents change)
so the UI can start quickly by loading the pre-built index.

Usage:
    python scripts/build_index.py [--force]
"""
from pathlib import Path
import argparse
import sys

# Project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_global_config
from src.models.rag_model import create_rag_model


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from documents")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild index even if it already exists",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=None,
        help="Documents directory (default: config documents_dir)",
    )
    args = parser.parse_args()

    config = get_global_config()
    docs_dir = args.documents or project_root / config.documents_dir

    if not docs_dir.exists():
        print(f"Documents directory not found: {docs_dir}")
        print("Create it and add .txt or .pdf files (e.g. 10-K filings).")
        sys.exit(1)

    pdf_files  = list(docs_dir.glob("*.pdf"))
    txt_files  = list(docs_dir.glob("*.txt"))
    file_count = len(pdf_files) + len(txt_files)

    if file_count == 0:
        print(f"No supported files (PDF, TXT) found in {docs_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s), {len(txt_files)} TXT(s) in {docs_dir}")
    print("Building index... (this may take several minutes on first run)")
    model = create_rag_model(docs_dir, force_reindex=args.force)
    stats = model.get_stats()
    print(f"Done. Index has {stats['num_nodes']} reference docs. Status: {stats['status']}")
    print("You can now start the UI with: streamlit run apps/rag_chat/app.py")


if __name__ == "__main__":
    main()
