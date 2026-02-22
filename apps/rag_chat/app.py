"""Streamlit chat interface for document Q&A (dataset-agnostic)."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.rag_model import create_rag_model
from src.safety.input_validator import InputValidator
from src.safety.output_filter import OutputFilter
from src.utils.logging import setup_logging

# Page config
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging(log_level="INFO")


@st.cache_resource
def load_rag_model():
    """Load RAG from pre-built index (fast). Build index first with: python scripts/build_index.py"""
    from src.models.rag_model import load_rag_from_index

    try:
        return load_rag_from_index()
    except FileNotFoundError:
        # Fallback: build from documents if index missing (slower)
        docs_dir = project_root / "data" / "documents"
        if docs_dir.exists() and any(docs_dir.iterdir()):
            return create_rag_model(docs_dir, force_reindex=False)
        return None


@st.cache_resource
def load_safety_components():
    """Load safety components (cached)"""
    return InputValidator(), OutputFilter()


def _normalize_confidence(value) -> float:
    """Normalize confidence to 0.0-1.0 for display."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0

    # Backend currently returns a 0-99 style score.
    if conf > 1.0:
        conf = conf / 100.0

    return max(0.0, min(conf, 1.0))


def _render_sources_and_confidence(
    sources,
    confidence,
    show_sources: bool,
    show_confidence: bool,
):
    """Render sources in collapsible section, then confidence as small text."""
    if show_sources:
        with st.expander("Sources", expanded=False):
            if sources:
                for source in sources:
                    file_name = source.get("file_name", "Unknown")
                    score = float(source.get("score", 0.0))
                    st.markdown(f"- `{file_name}` (score: {score:.3f})")
            else:
                st.markdown("- No sources retrieved")

    if show_confidence:
        conf = _normalize_confidence(confidence)
        st.caption(f"Confidence: {conf:.0%}")


def main():
    st.title("Document Q&A Chatbot")

    # Generic description - works for any dataset
    st.markdown("""
    ### About

    This **Q&A chatbot** answers questions using the documents in your index. The dataset is
    configurable: add PDFs or text files to `data/documents/`, run `python scripts/build_index.py`,
    then ask questions here.

    **Features:** source citations, optional safety checks, confidence scores.
    """)

    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        safety_enabled = st.checkbox("Enable Safety Checks", value=True)
        show_sources = st.checkbox("Show Source Documents", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)

        st.markdown("---")
        st.markdown("### About")
        st.info("""
        Document Q&A over your indexed corpus.

        - Source citations
        - Safety filtering
        - Confidence indicators
        """)

    # Load model
    with st.spinner("Loading RAG system..."):
        rag_model = load_rag_model()

    if rag_model is None:
        st.error("""
        **No index found.**

        Add documents to `data/documents/`, then build the index:
        ```bash
        python scripts/build_index.py
        ```

        Optional: `python scripts/download_sec_filings.py` for sample SEC 10-K data.
        """)
        return

    # Load safety components if enabled
    if safety_enabled:
        input_validator, output_filter = load_safety_components()
    else:
        input_validator, output_filter = None, None

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message or "confidence" in message:
                _render_sources_and_confidence(
                    message.get("sources", []),
                    message.get("confidence", 0.0),
                    show_sources=show_sources,
                    show_confidence=show_confidence,
                )

            # Show safety warnings if any
            if "safety_warning" in message:
                st.warning(message["safety_warning"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query with or without safety
                    if safety_enabled:
                        result = rag_model.query_with_safety(
                            prompt,
                            input_validator,
                            output_filter
                        )
                    else:
                        result = rag_model.query(prompt)

                    answer = result["answer"]
                    sources = result.get("source_nodes", [])
                    confidence = result.get("metadata", {}).get("confidence", 0.0)
                    blocked = result.get("metadata", {}).get("blocked", False)

                    # Display answer
                    st.markdown(answer)
                    _render_sources_and_confidence(
                        sources,
                        confidence,
                        show_sources=show_sources,
                        show_confidence=show_confidence,
                    )

                    # Build message for history
                    message_data = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "confidence": confidence,
                    }

                    # Show safety warnings
                    if blocked:
                        warning = f"Warning: Query was blocked: {result.get('metadata', {}).get('reason', 'Safety check failed')}"
                        st.warning(warning)
                        message_data["safety_warning"] = warning

                    # Add to history
                    st.session_state.messages.append(message_data)

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
