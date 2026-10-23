"""Business-knowledge RAG built on LangChain + LangGraph.

The retrieval pipeline is fully offline and deterministic — no LLM, no
embedding API, no vector store:

    markdown docs
      → ``MarkdownHeaderTextSplitter`` (langchain-text-splitters): every
        ``##``/``###`` heading becomes a chunk whose title inherits its
        parents, e.g. ``指标口径 > GMV（成交总额）``
      → ``TitleBodyRetriever`` (a ``langchain_core`` ``BaseRetriever``):
        TF-IDF over Chinese character bigrams + ASCII words, with title and
        body scored as separate length-normalised fields (2:1) so a long
        body cannot dilute a decisive title match
      → a LangGraph ``StateGraph``: ``retrieve → grade → compose``

When langchain / langgraph are not installed the module falls back to an
equivalent pure-Python path (same tokenizer, same scoring, same public
API), so the agent keeps working with zero third-party agent frameworks.

Usage::

    kb = KnowledgeBase()                 # loads docs/ by default
    hits = kb.search("GMV 的口径是什么")  # ranked chunks
    prompt = kb.context_for("feed 渠道")  # formatted context block
    out = kb.answer("GMV 的口径是什么")   # full RAG run through the graph
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Optional LangChain / LangGraph stack (preferred path when importable).
# --------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly by the langchain tests
    from typing import TypedDict

    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    from langgraph.graph import END, START, StateGraph

    HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover - fallback path
    HAS_LANGCHAIN = False

_DOC_DIR = Path(__file__).resolve().parents[1] / "docs"

_CJK = r"\u4e00-\u9fff"


def _tokenize(text: str) -> List[str]:
    """Chinese character bigrams + single chars, plus ASCII words.

    Bigrams carry most of the Chinese semantics ("口径", "退款"); single chars
    keep short queries matchable. ASCII words handle tables/column names.
    """
    tokens: List[str] = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        tokens.append(word.lower())
    for run in re.findall(rf"[{_CJK}]+", text):
        if len(run) == 1:
            tokens.append(run)
            continue
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
        tokens.extend(run)  # single chars as weak extra evidence
    return tokens


class _Indexed:
    """Internal record: a chunk plus its normalised title/body TF vectors."""

    __slots__ = ("title", "text", "source", "title_vec", "body_vec")

    def __init__(self, title: str, text: str, source: str):
        self.title = title
        self.text = text
        self.source = source
        self.title_vec: Dict[str, float] = {}
        self.body_vec: Dict[str, float] = {}


def _chunk_markdown(path: Path) -> List[Tuple[str, str, str]]:
    """Pure-Python chunking: split on ##/### headings; children inherit parents.

    Only used on the fallback path (LangChain's ``MarkdownHeaderTextSplitter``
    does the equivalent job when available). Returns (title, body, source).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: List[Tuple[str, str, str]] = []
    parents: List[str] = []
    current: List[str] = []
    title = ""

    def flush() -> None:
        nonlocal current, title
        body = "\n".join(current).strip()
        if title and body:
            chunks.append((title, body, path.name))
        current = []

    for line in lines:
        if line.startswith("## "):
            flush()
            parents = [line[3:].strip()]
            title = parents[0]
        elif line.startswith("### "):
            flush()
            title = " > ".join(parents + [line[4:].strip()])
        elif line.strip() == "---":
            flush()
        else:
            current.append(line)
    flush()
    # Drop the file preamble (before the first ##): it has no topic title.
    return [c for c in chunks if c[0] and " > " in c[0]]


def _chunk_markdown_langchain(path: Path) -> List[Tuple[str, str, str]]:
    """LangChain chunking via ``MarkdownHeaderTextSplitter``.

    Every ``##``/``###`` heading becomes metadata (``h2``/``h3``); the chunk
    title joins the header chain with `` > `` exactly like the fallback.
    """
    splitter = MarkdownHeaderTextSplitter(
        # h2/h3 only: the file's top-level `#` title is boilerplate, not part
        # of any chunk title, so `# 业务知识库` never leaks into retrieval.
        headers_to_split_on=[("##", "h2"), ("###", "h3")],
        strip_headers=True,
    )
    out: List[Tuple[str, str, str]] = []
    for doc in splitter.split_text(path.read_text(encoding="utf-8")):
        meta = doc.metadata
        parts = [meta.get(k, "").strip() for k in ("h2", "h3")]
        title = " > ".join(p for p in parts if p)
        body = doc.page_content.strip()
        # Skip the preamble (no heading) and title-less fragments.
        if title and body and " > " in title:
            out.append((title, body, path.name))
    return out


def _build_idf(indexed: List[_Indexed]) -> Dict[str, float]:
    """Document frequency → idf over the whole corpus."""
    df: Dict[str, int] = {}
    for rec in indexed:
        for vec in (rec.title_vec, rec.body_vec):
            for tok in vec:
                df[tok] = df.get(tok, 0) + 1
    n = len(indexed) or 1
    return {tok: math.log((n + 1) / (cnt + 0.5)) for tok, cnt in df.items()}


def _norm(tf: Dict[str, int]) -> Dict[str, float]:
    norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
    return {tok: cnt / norm for tok, cnt in tf.items()}


def _score_all(query: str, indexed: List[_Indexed], idf: Dict[str, float],
               k: int, min_score: float) -> List[Dict[str, object]]:
    """Shared scoring: 2 * cosine(query, title) + cosine(query, body), idf-weighted."""
    qtf: Dict[str, int] = {}
    for tok in _tokenize(query):
        qtf[tok] = qtf.get(tok, 0) + 1
    qnorm = math.sqrt(sum(v * v for v in qtf.values())) or 1.0
    qvec = {tok: v / qnorm for tok, v in qtf.items()}

    def _cos(vec: Dict[str, float]) -> float:
        dot = 0.0
        for tok, qv in qvec.items():
            cv = vec.get(tok)
            if cv is not None:
                dot += qv * cv * idf.get(tok, 1.0)
        return dot

    scored: List[Tuple[float, _Indexed]] = []
    for rec in indexed:
        score = 2.0 * _cos(rec.title_vec) + _cos(rec.body_vec)
        if score >= min_score:
            scored.append((score, rec))
    scored.sort(key=lambda pair: -pair[0])
    return [
        {"title": rec.title, "text": rec.text, "source": rec.source,
         "score": round(float(s), 4)}
        for s, rec in scored[:k]
    ]


def _format_context(hits: List[Dict[str, object]]) -> str:
    """Prompt-ready context block shared by both paths (empty if no hit)."""
    if not hits:
        return ""
    parts = []
    for hit in hits:
        body = re.sub(r"\n{2,}", "\n", str(hit["text"])).strip()
        if len(body) > 380:
            body = body[:380] + "…"
        parts.append(f"【{hit['title']}】{body}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# LangChain retriever + LangGraph pipeline (preferred path).
# --------------------------------------------------------------------------
if HAS_LANGCHAIN:

    class TitleBodyRetriever(BaseRetriever):
        """LangChain retriever with field-weighted TF-IDF scoring.

        Title and body are scored as separate length-normalised fields (2:1)
        so a decisive title match is not diluted by a long body. Deterministic
        and offline: plain cosine over idf-weighted TF vectors, no embeddings.
        """

        records: list  # list[_Indexed]
        idf: dict      # token -> idf weight
        k: int = 3
        min_score: float = 0.12

        def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs) -> List[Document]:
            hits = _score_all(query, list(self.records), dict(self.idf), self.k, self.min_score)
            return [
                Document(
                    page_content=str(hit["text"]),
                    metadata={"title": hit["title"], "source": hit["source"],
                              "score": hit["score"]},
                )
                for hit in hits
            ]

    class _RagState(TypedDict):
        """Graph state schema: separate channels, merged across nodes."""

        query: str
        docs: list
        context: str

    def _make_rag_graph(retriever: TitleBodyRetriever, k: int):
        """Build the RAG StateGraph: retrieve → grade → compose.

        - retrieve: TitleBodyRetriever fetches top-k chunks
        - grade: drop chunks under the relevance threshold (belt and braces —
          the retriever already applies it, so grading is observable in the
          state and cheap to re-tune without touching retrieval)
        - compose: format the graded chunks into a prompt-ready context block
        """

        def retrieve(state: dict) -> dict:
            docs = retriever.invoke(state["query"])
            return {"docs": docs}

        def grade(state: dict) -> dict:
            docs = [d for d in state["docs"] if d.metadata.get("score", 0.0) >= retriever.min_score]
            return {"docs": docs}

        def compose(state: dict) -> dict:
            hits = [
                {"title": d.metadata["title"], "text": d.page_content,
                 "source": d.metadata["source"]}
                for d in state["docs"]
            ]
            return {"context": _format_context(hits)}

        graph = StateGraph(_RagState)
        graph.add_node("retrieve", retrieve)
        graph.add_node("grade", grade)
        graph.add_node("compose", compose)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_edge("grade", "compose")
        graph.add_edge("compose", END)
        return graph.compile()


class KnowledgeBase:
    """Business-knowledge RAG: LangChain chunking + TF-IDF retriever, LangGraph flow.

    Falls back to the pure-Python path (identical API and scoring) when
    langchain/langgraph are not installed.
    """

    engine = "langchain+langgraph" if HAS_LANGCHAIN else "pure-python fallback"

    def __init__(self, doc_dir: Optional[Path] = None, files: Optional[List[str]] = None):
        if files:
            paths = [Path(doc_dir or _DOC_DIR) / name for name in files]
        elif doc_dir is not None:
            # Explicit directory: take every markdown file in it.
            paths = sorted(Path(doc_dir).glob("*.md"))
        else:
            # Default: the business-knowledge corpus plus the metric-definition
            # knowledge base, both in docs/.
            paths = sorted(
                p
                for p in _DOC_DIR.glob("*.md")
                if "business_knowledge" in p.name or "knowledge" in p.name or "metric_dictionary" in p.name
            )

        triplets: List[Tuple[str, str, str]] = []
        for path in paths:
            if not path.exists():
                continue
            triplets.extend(_chunk_markdown_langchain(path) if HAS_LANGCHAIN else _chunk_markdown(path))

        self.chunks: List[_Indexed] = [_Indexed(t, b, s) for t, b, s in triplets]
        for rec in self.chunks:
            title_tf: Dict[str, int] = {}
            body_tf: Dict[str, int] = {}
            for tok in _tokenize(rec.title):
                title_tf[tok] = title_tf.get(tok, 0) + 1
            for tok in _tokenize(rec.text):
                body_tf[tok] = body_tf.get(tok, 0) + 1
            rec.title_vec = _norm(title_tf)
            rec.body_vec = _norm(body_tf)
        self._idf: Dict[str, float] = _build_idf(self.chunks)

        # LangChain-native accessors (None on the fallback path).
        self.retriever = None
        self.graph = None
        if HAS_LANGCHAIN and self.chunks:
            self.retriever = TitleBodyRetriever(
                records=self.chunks, idf=self._idf, k=3, min_score=0.12
            )
            self.graph = _make_rag_graph(self.retriever, k=3)

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 3, min_score: float = 0.12) -> List[Dict[str, object]]:
        """Top-k chunks: 2 * cosine(query, title) + cosine(query, body), idf-weighted."""
        return _score_all(query, self.chunks, self._idf, k, min_score)

    def retrieve(self, query: str, k: int = 3) -> List["Document"]:
        """LangChain-native retrieval: returns ``Document`` objects (or [] without langchain)."""
        if self.retriever is None:
            return []
        self.retriever.k = k
        return list(self.retriever.invoke(query))

    def context_for(self, query: str, k: int = 3) -> str:
        """Formatted context block for narratives / prompts (empty if no hit)."""
        return _format_context(self.search(query, k=k))

    def answer(self, query: str, k: int = 3) -> Dict[str, object]:
        """Run the full RAG pipeline for one query.

        On the LangChain path this invokes the LangGraph state machine
        (retrieve → grade → compose); the fallback composes directly.
        """
        if self.graph is not None:
            state = self.graph.invoke({"query": query})
            docs = state.get("docs", [])
            return {
                "query": query,
                "context": state.get("context", ""),
                "sources": [d.metadata["title"] for d in docs],
                "scores": [d.metadata.get("score") for d in docs],
                "engine": "langgraph",
            }
        hits = self.search(query, k=k)
        return {
            "query": query,
            "context": _format_context(hits),
            "sources": [h["title"] for h in hits],
            "scores": [h["score"] for h in hits],
            "engine": "fallback",
        }

    def __len__(self) -> int:
        return len(self.chunks)
