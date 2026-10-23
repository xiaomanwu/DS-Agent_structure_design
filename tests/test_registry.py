"""Runtime extension registry: register, route, dispatch, catalog, cleanup."""

from __future__ import annotations

import pandas as pd
import pytest

from ds_agent.agent import DSQueryAgent, full_catalog
from ds_agent.registry import (
    Extension,
    clear_extensions,
    extension_catalog,
    extension_handlers,
    extension_scores,
    register_extension,
    unregister_extension,
)
from ds_agent.reply import AgentReply


@pytest.fixture(autouse=True)
def clean_registry():
    clear_extensions()
    yield
    clear_extensions()


def upper_handler(question: str, slots: dict) -> AgentReply:
    return AgentReply(
        text=f"ECHO: {question.upper()}",
        tool_calls=[{"tool": "echo", "args": {}, "result": {"ok": True}}],
        data={"intent": "echo_ext"},
    )


def test_register_validates_name_and_handler():
    with pytest.raises(ValueError):
        register_extension(Extension(name="BadName", handler=upper_handler))
    with pytest.raises(TypeError):
        register_extension(Extension(name="ok_name", handler="not callable"))
    ext = register_extension(Extension(name="echo_ext", handler=upper_handler, keywords=["回声|echo"]))
    assert ext.catalog_entry()["zh"] == "echo_ext"  # metadata defaults to the name


def test_extension_routes_and_dispatches(warehouse, workdir):
    register_extension(
        Extension(
            name="echo_ext",
            handler=upper_handler,
            keywords=[r"回声测试|echo test"],
            zh="回声扩展",
            en="Echo extension",
            example_zh="做个回声测试",
            example_en="run an echo test",
        )
    )
    agent = DSQueryAgent(warehouse=warehouse, artifact_dir=workdir / "artifacts_ext")
    reply = agent.ask("做个回声测试")
    assert reply.data["intent"] == "echo_ext"
    assert reply.text.startswith("ECHO:")
    agent.close()


def test_extension_joins_catalogs_and_help(warehouse, workdir):
    register_extension(
        Extension(name="echo_ext", handler=upper_handler, keywords=[r"echo"], zh="回声扩展", en="Echo")
    )
    assert "echo_ext" in full_catalog()
    assert "echo_ext" in extension_catalog()
    assert "echo_ext" in extension_handlers()
    agent = DSQueryAgent(warehouse=warehouse, artifact_dir=workdir / "artifacts_ext")
    assert "echo_ext" in agent._help().text
    agent.close()


def test_extension_scores_match_keywords():
    register_extension(Extension(name="a_ext", handler=upper_handler, keywords=[r"波峰|spike"], weight=9))
    register_extension(Extension(name="b_ext", handler=upper_handler, keywords=[r"绝不匹配"]))
    scores = extension_scores("帮我看看这个波峰是什么")
    assert scores == {"a_ext": 9}
    assert extension_scores("无关问题") == {}


def test_unregister_and_clear():
    register_extension(Extension(name="echo_ext", handler=upper_handler))
    assert unregister_extension("echo_ext") is True
    assert unregister_extension("echo_ext") is False
    assert extension_catalog() == {}


def test_agent_register_extension_convenience(warehouse, workdir):
    agent = DSQueryAgent(warehouse=warehouse, artifact_dir=workdir / "artifacts_ext")
    agent.register_extension(
        name="echo_ext", handler=upper_handler, keywords=[r"回声测试"], zh="回声", en="Echo"
    )
    reply = agent.ask("做个回声测试")
    assert reply.data["intent"] == "echo_ext"
    agent.close()
