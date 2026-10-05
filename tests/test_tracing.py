"""Tests for the optional LangSmith tracing.

The point of tracing here is to tell the workflow's traces from the agent's, so
what is pinned is the tagging and the off-by-default behaviour - not whether
LangChain's client works.
"""

from __future__ import annotations

import pytest

from rr import settings, tracing


@pytest.fixture
def off(monkeypatch):
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", False)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setattr(settings, "LANGSMITH_PROJECT", "route-or-roam")


def test_off_by_default_returns_an_empty_config(off):
    """Callers merge it unconditionally, so it must change nothing when off."""
    assert tracing.run_config("workflow", "r1", "q1") == {}
    assert tracing.status()["enabled"] is False


def test_a_key_alone_does_not_switch_it_on(monkeypatch):
    """A LANGSMITH_API_KEY left in the environment by another project must not
    start shipping this project's questions and answers somewhere."""
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", False)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "ls-test")
    assert tracing.run_config("agent", "r1", "q1") == {}


def test_asked_for_but_unkeyed_is_named(monkeypatch):
    """The dangerous state: a run you believed was traced produced no evidence."""
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "")
    status = tracing.status()
    assert status["enabled"] is False
    assert "nothing is recorded" in status["note"]


def test_each_system_is_tagged_so_the_two_can_be_compared(on):
    workflow = tracing.run_config("workflow", "run-7", "q3")
    agent = tracing.run_config("agent", "run-7", "q3")
    assert "workflow" in workflow["tags"] and "agent" in agent["tags"]
    # Same run and question on both sides: that pairing is what makes the two
    # traces comparable question by question.
    assert workflow["metadata"]["qid"] == agent["metadata"]["qid"] == "q3"
    assert workflow["metadata"]["run_id"] == agent["metadata"]["run_id"] == "run-7"
    assert workflow["run_name"] != agent["run_name"]


def test_enabled_status_states_what_leaves_the_machine(on):
    """The privacy cost travels with the status, so it reaches the README."""
    assert "sent to LangChain" in tracing.status()["note"]
