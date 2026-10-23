"""RAG knowledge base: chunking, retrieval relevance, and agent wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from ds_agent.knowledge import HAS_LANGCHAIN, KnowledgeBase, _tokenize

_requires_langchain = pytest.mark.skipif(
    not HAS_LANGCHAIN, reason="langchain/langgraph not installed"
)


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def test_corpus_is_chunked_by_headings(kb):
    assert len(kb) >= 15
    titles = {c.title for c in kb.chunks}
    assert any("GMV" in t for t in titles)
    assert any("埋点中断" in t for t in titles)
    # Every chunk inherits its parent heading ("parent > child").
    assert all(" > " in c.title for c in kb.chunks)


def test_tokenizer_mixes_bigrams_and_words():
    tokens = _tokenize("GMV 口径 refund_rate 退款")
    assert "gmv" in tokens
    assert "refund_rate" in tokens
    assert "口径" in tokens  # bigram
    assert "退款" in tokens


def test_retrieval_ranks_caliber_questions_correctly(kb):
    # A terse caliber query must surface the caliber chunk itself.
    hits = kb.search("GMV 口径", k=2)
    assert hits[0]["title"] == "指标口径 > GMV（成交总额）"
    # A natural-language question may prefer the FAQ entry that answers it,
    # but GMV-related content must dominate the top-3 either way.
    hits = kb.search("GMV 的口径是什么，和 revenue 有什么区别", k=3)
    top_titles = " ".join(h["title"] for h in hits)
    assert "GMV" in top_titles
    assert any("指标口径" in h["title"] or "FAQ" in h["title"] for h in hits)


def test_retrieval_finds_known_incident_by_date(kb):
    hits = kb.search("2026-10-15 feed 埋点中断", k=2)
    assert hits
    assert "埋点中断" in hits[0]["title"]


def test_context_for_is_prompt_ready(kb):
    ctx = kb.context_for("为什么转化率涨了但 GMV 跌了", k=2)
    assert ctx.startswith("【")
    assert "转化率" in ctx
    assert len(ctx) < 1200


def test_no_match_returns_empty(kb):
    assert kb.search("量子纠缠叠加态xxqq", k=3) == []
    assert kb.context_for("量子纠缠叠加态xxqq") == ""


def test_custom_dir_and_missing_files_are_tolerated(tmp_path_factory):
    # pytest's default tmp base is not writable in this sandbox; use the repo.
    tmp_dir = Path(__file__).parent / ".tmp" / "kb_custom"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for stale in tmp_dir.glob("*.md"):
        stale.unlink()
    empty = KnowledgeBase(doc_dir=tmp_dir)
    assert len(empty) == 0
    assert empty.search("anything") == []
    (tmp_dir / "k.md").write_text("## A\n\n### B\nbody text 口径\n", encoding="utf-8")
    small = KnowledgeBase(doc_dir=tmp_dir)
    assert len(small) == 1
    assert small.search("口径")[0]["title"] == "A > B"


def test_agent_answers_from_knowledge_base(agent):
    reply = agent.ask("GMV 的口径是什么？和 revenue 有什么区别？")
    assert agent.route("GMV 的口径是什么")[0] == "knowledge"
    assert "知识库" in reply.text or "knowledge" in reply.text.lower()
    assert "GMV" in reply.text
    assert "知识条目" in reply.text or any(b["type"] == "table" for b in reply.blocks)
    assert reply.data["knowledge"]["hits"]


def test_sql_reply_carries_caliber_note(agent):
    reply = agent.ask("近7天各渠道 GMV 排名")
    assert "业务口径" in reply.text
    assert "未支付" in reply.text  # the caliber note, not just the numbers


# ---------------------------------------------------------------- LangChain


@_requires_langchain
def test_langchain_stack_is_active(kb):
    assert kb.engine == "langchain+langgraph"
    assert kb.retriever is not None
    assert kb.graph is not None


@_requires_langchain
def test_langchain_retriever_returns_documents(kb):
    from langchain_core.documents import Document

    docs = kb.retrieve("GMV 口径", k=2)
    assert docs
    assert all(isinstance(d, Document) for d in docs)
    assert "GMV" in docs[0].metadata["title"]
    assert docs[0].metadata["score"] >= 0.12


@_requires_langchain
def test_langgraph_answer_pipeline(kb):
    out = kb.answer("GMV 的口径是什么")
    assert out["engine"] == "langgraph"
    assert out["context"].startswith("【")
    assert out["sources"]
    assert all(s >= 0.12 for s in out["scores"])
    # Empty KB must not break the graph contract.
    empty = KnowledgeBase(doc_dir=Path(__file__).parent / ".tmp" / "kb_graph_empty")
    assert empty.answer("anything")["engine"] == "fallback"
    assert empty.answer("anything")["context"] == ""
