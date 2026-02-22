"""
OpenAI Batch Embeddings – 50% cheaper than the standard Embeddings API.

The OpenAI Batch API processes embedding requests asynchronously with a
24-hour turnaround window and charges 50% of the synchronous price.

  Standard API:  $0.020 / 1M tokens (text-embedding-3-small)
  Batch API:     $0.010 / 1M tokens

Use this script INSTEAD of `build_index.py` when you want maximum savings
and can tolerate the ~5-30 minute wait for batch completion.

Workflow:
  1. Load + preprocess documents (SEC noise stripped)
  2. Chunk into 512-token pieces
  3. Write a JSONL file (one embedding request per chunk)
  4. Upload JSONL -> submit batch -> poll until done
  5. Download results -> store in ChromaDB
  6. Index is ready for the Streamlit UI

Usage:
    python scripts/embed_batch.py
    python scripts/embed_batch.py --docs-dir data/documents --model text-embedding-3-small
    python scripts/embed_batch.py --poll-only --batch-id batch_abc123  # resume polling

Requirements:
    OPENAI_API_KEY must be set in .env
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from repo root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.rag.document_loader import DocumentLoader
from src.utils.config import get_global_config
from src.utils.logging import get_logger, setup_logging

# Configure logging immediately so every logger.info/error prints to console.
setup_logging(log_level="INFO")
logger = get_logger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data" / "documents"
BATCH_DIR = PROJECT_ROOT / "data" / "processed" / "batch_embeddings"


def build_jsonl(chunks, model: str, jsonl_path: Path) -> None:
    """Write one embedding request per chunk to a JSONL file."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            request = {
                "custom_id": f"chunk-{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model,
                    "input": chunk.text,
                    "encoding_format": "float",
                },
            }
            f.write(json.dumps(request) + "\n")
    logger.info(f"Wrote {len(chunks)} requests -> {jsonl_path}")


def submit_batch(client, jsonl_path: Path) -> str:
    """Upload JSONL file and submit batch job. Returns batch ID."""
    logger.info("Uploading JSONL file to OpenAI Files API...")
    with open(jsonl_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    logger.info(f"File uploaded: {upload.id}")

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
    )
    logger.info(f"Batch submitted: {batch.id}  (status: {batch.status})")
    return batch.id


def poll_batch(client, batch_id: str, poll_interval: int = 30) -> str:
    """Poll until batch completes. Returns output_file_id."""
    logger.info(f"Polling batch {batch_id} every {poll_interval}s ...")
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        logger.info(
            f"  status={batch.status}  "
            f"completed={counts.completed}/{counts.total}  "
            f"failed={counts.failed}"
        )
        if batch.status == "completed":
            return batch.output_file_id
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch ended with status: {batch.status}")
        time.sleep(poll_interval)


def download_results(client, output_file_id: str, output_path: Path) -> None:
    """Download batch output JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(output_file_id)
    output_path.write_bytes(content.read())
    logger.info(f"Results downloaded -> {output_path}")


def store_in_chroma(chunks, results_path: Path, config) -> None:
    """Load embeddings from batch output and insert into ChromaDB."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    persist_dir = str(config.rag.vector_store.persist_directory)
    collection_name = config.rag.vector_store.collection_name

    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Parse results
    embeddings_map: dict[str, list[float]] = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            if result.get("error"):
                logger.warning(f"Embedding failed for {custom_id}: {result['error']}")
                continue
            vector = result["response"]["body"]["data"][0]["embedding"]
            embeddings_map[custom_id] = vector

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        key = f"chunk-{i}"
        if key not in embeddings_map:
            logger.warning(f"No embedding for {key}, skipping")
            continue
        ids.append(key)
        embeddings.append(embeddings_map[key])
        documents.append(chunk.text)
        metadatas.append({k: str(v) for k, v in chunk.metadata.items()})

    # ChromaDB upsert in batches of 500
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(f"Stored {len(ids)} chunks in ChromaDB collection '{collection_name}'")


def inspect_chroma(config) -> None:
    """Print a summary of what is currently stored in ChromaDB."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    persist_dir = str(config.rag.vector_store.persist_directory)
    collection_name = config.rag.vector_store.collection_name

    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name not in existing:
        print(f"\nCollection '{collection_name}' does not exist yet.")
        print(f"  Persist dir : {persist_dir}")
        print("  No embeddings stored. Run embed_batch.py (without --inspect) first.")
        return

    col = chroma_client.get_collection(collection_name)
    count = col.count()

    print(f"\n{'='*60}")
    print(f"  ChromaDB collection : {collection_name}")
    print(f"  Persist dir         : {persist_dir}")
    print(f"  Total chunks stored : {count:,}")
    print(f"{'='*60}")

    if count == 0:
        print("  Collection is empty.")
        return

    # Sample a few entries to show sources
    sample = col.peek(min(10, count))
    sources = {}
    for meta in sample.get("metadatas", []):
        src = meta.get("file_name") or meta.get("source") or "unknown"
        sources[src] = sources.get(src, 0) + 1

    print("  Sample sources (first 10 chunks):")
    for src, n in sources.items():
        print(f"    {src}  ({n} chunk(s) in sample)")

    # Full source breakdown via get()
    print("\n  Full source breakdown:")
    try:
        all_metas = col.get(include=["metadatas"])["metadatas"]
        breakdown: dict = {}
        for m in all_metas:
            src = m.get("file_name") or m.get("source") or "unknown"
            breakdown[src] = breakdown.get(src, 0) + 1
        for src, n in sorted(breakdown.items()):
            print(f"    {n:>5}  chunks  <-  {src}")
    except Exception as e:
        print(f"  (Could not fetch full breakdown: {e})")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch embedding via OpenAI Batch API (50% cheaper)")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="Directory with .txt/.pdf files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status polls")
    parser.add_argument("--poll-only", action="store_true", help="Skip submission, poll an existing batch")
    parser.add_argument("--batch-id", default=None, help="Batch ID to resume polling (use with --poll-only)")
    parser.add_argument("--dry-run", action="store_true", help="Build JSONL only, do not submit")
    parser.add_argument("--inspect", action="store_true", help="Show what is stored in ChromaDB and exit")
    args = parser.parse_args()

    config = get_global_config()

    # ── Inspect mode: show ChromaDB contents and exit ─────────────────────────
    if args.inspect:
        inspect_chroma(config)
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.error("OPENAI_API_KEY is not set. Add it to .env.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = BATCH_DIR / "requests.jsonl"
    results_path = BATCH_DIR / "results.jsonl"

    if not args.poll_only:
        # ── Load, preprocess, and chunk documents ────────────────────────────
        # DocumentLoader.load_documents() already runs chunking internally.
        docs_dir = Path(args.docs_dir)
        loader = DocumentLoader(config.rag.chunking)
        chunks = loader.load_documents(docs_dir)
        if not chunks:
            logger.error(f"No documents found in {docs_dir}")
            sys.exit(1)
        logger.info(f"Loaded and chunked -> {len(chunks)} chunk(s)")

        # ── Estimate cost ─────────────────────────────────────────────────────
        total_chars = sum(len(c.text) for c in chunks)
        estimated_tokens = total_chars // 4
        # Batch API price for text-embedding-3-small = $0.010 / 1M tokens
        estimated_cost = estimated_tokens / 1_000_000 * 0.010
        logger.info(
            f"Estimated tokens: {estimated_tokens:,}  "
            f"Estimated batch cost: ${estimated_cost:.4f} "
            f"(50% saving vs standard API)"
        )

        build_jsonl(chunks, args.model, jsonl_path)

        if args.dry_run:
            logger.info(f"--dry-run: JSONL written to {jsonl_path}, not submitted.")
            return

        batch_id = submit_batch(client, jsonl_path)
        # Save batch_id so --poll-only can resume
        (BATCH_DIR / "batch_id.txt").write_text(batch_id)
    else:
        batch_id = args.batch_id
        if not batch_id:
            id_file = BATCH_DIR / "batch_id.txt"
            if id_file.exists():
                batch_id = id_file.read_text().strip()
            else:
                logger.error("Provide --batch-id or run without --poll-only first.")
                sys.exit(1)

        # Reload chunks so we can store them alongside embeddings
        docs_dir = Path(args.docs_dir)
        loader = DocumentLoader(config.rag.chunking)
        chunks = loader.load_documents(docs_dir)

    # ── Poll and store ────────────────────────────────────────────────────────
    output_file_id = poll_batch(client, batch_id, poll_interval=args.poll_interval)
    download_results(client, output_file_id, results_path)
    store_in_chroma(chunks, results_path, config)

    logger.info("Done. Start the UI with:")
    logger.info("  streamlit run apps/rag_chat/app.py")


if __name__ == "__main__":
    main()
