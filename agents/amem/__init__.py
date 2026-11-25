"""A-MEM vendored modules.

Source: agiresearch/A-mem @ commit ceffb860f071
Paper: arXiv 2502.12110 (Xu et al. 2025)
License: MIT (see external/A-mem/LICENSE)

Files in this package:
    memory_system.py  — Verbatim copy of upstream agentic_memory/memory_system.py
                        (MemoryNote, AgenticMemorySystem core)
    retrievers.py     — Verbatim copy of upstream agentic_memory/retrievers.py
                        (ChromaRetriever wrapper)
    llm_controller.py — Patched: adds base_url support so we can route to
                        DeepSeek (or any OpenAI-compatible endpoint) via
                        existing HYMEM_*_MODEL env vars without introducing
                        new API keys.

Lazy-import strategy: the heavy modules (chromadb / sentence-transformers /
nltk) are only loaded when explicitly accessed. This lets non-A-MEM code
paths and lightweight unit tests import this package without paying the
full dependency chain.
"""

# Heavy imports are intentionally NOT done at package import time.
# Use:
#     from agents.amem.memory_system import AgenticMemorySystem
#     from agents.amem.llm_controller import LLMController
# inside the call site that actually needs them (see agents/amem_agent.py).

__all__ = ["AgenticMemorySystem", "MemoryNote", "ChromaRetriever", "LLMController"]


def __getattr__(name):
    """PEP 562: lazy module-level attributes. Only triggers heavy imports
    when someone does `from agents.amem import AgenticMemorySystem` etc."""
    if name in ("AgenticMemorySystem", "MemoryNote"):
        from .memory_system import AgenticMemorySystem, MemoryNote
        return locals()[name]
    if name == "ChromaRetriever":
        from .retrievers import ChromaRetriever
        return ChromaRetriever
    if name == "LLMController":
        from .llm_controller import LLMController
        return LLMController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
