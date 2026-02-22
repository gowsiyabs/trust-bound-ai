"""Prompt templates for query engine.

Kept intentionally tight: every extra token in the prompt costs money at
generation time. The context window is filled by the retrieved chunks;
the instructions are a small, fixed overhead.
"""

# Primary Q&A prompt. gpt-4o-mini handles this well at 2-4 sentence answers.
DOCUMENT_QA_PROMPT_TEMPLATE = """\
You are a financial document assistant. Answer using ONLY the context below.
Be concise: 2-4 sentences unless a list is clearly needed.
If the answer is not in the context, reply: "Not found in the provided documents."

Context:
{context_str}

Question: {query_str}
Answer:"""


# Query rewrite prompt – used by the hybrid retriever for better recall.
QUERY_REWRITE_PROMPT_TEMPLATE = """\
Rewrite the query below to improve document retrieval. \
Add specific financial or company keywords if relevant. \
Return only the rewritten query, nothing else.

Query: {query}
Rewritten:"""


# Hallucination check – used by the output filter.
HALLUCINATION_CHECK_PROMPT = """\
Is the answer below fully supported by the context?
Answer YES or NO, then one sentence explanation.

Context: {context}
Question: {question}
Answer: {answer}

Supported?"""
