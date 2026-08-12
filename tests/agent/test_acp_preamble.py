"""The preamble that stops the model narrating instead of acting.

Why these tests exist: the ACP prompt once ended with "If no tool is needed,
answer normally." Measured on a long Opus 5 session, 5 of 5 ending turns said
some form of "I will now check X" and then stopped — the agent loop halted, and
the operator reported the session as dead. The work was correct; the turn handed
control back before doing it.

Commit 49fc1d3a0 replaced that line with two instructions: prose is for an
answer/question/blocker (not for narrating a next step), and multi-step work
should keep calling tools until done. Mutation testing found that the fix had
NO test coverage at all — both mutations that reverted it were MISSED by the
22-test suite. These are those missing tests: each one fails if the phrasing
that caused the stall comes back.
"""
import pytest

from agent import copilot_acp_client as acp


def _preamble(**kw):
    """The prompt for a trivial one-message conversation."""
    return acp._format_messages_as_prompt([{"role": "user", "content": "hi"}], **kw)


def test_the_blanket_prose_permission_is_gone():
    """The exact sentence that read as permission to reply instead of act.

    "If no tool is needed, answer normally" is true but useless here: a turn that
    describes its own next step believes no tool is needed *yet*, so the sentence
    licensed the stall.
    """
    prompt = _preamble()
    assert "If no tool is needed, answer normally" not in prompt


def test_prose_is_scoped_to_answer_question_or_blocker():
    """Prose stays allowed — the fix narrows when, it does not forbid."""
    prompt = _preamble()
    lowered = prompt.lower()
    assert "prose is for an answer" in lowered
    for allowed in ("question", "blocker"):
        assert allowed in lowered, f"prose must still be allowed for a {allowed}"


def test_narrating_a_next_step_is_ruled_out_explicitly():
    """The instruction must name the failure, not gesture at it.

    A vague "be proactive" does not survive a long session; the measured stall
    was a specific shape — announce, then end the turn — so the prompt names
    that shape and its consequence.
    """
    prompt = _preamble()
    lowered = prompt.lower()
    assert "narrating what you are about to do" in lowered
    assert "in this turn" in lowered, "must say to emit the call in the SAME turn"
    assert "halts the loop" in lowered, "must state the consequence of ending early"


def test_multi_step_work_is_asked_for_in_one_turn():
    """Without this, the model reports after every single step."""
    prompt = _preamble()
    lowered = prompt.lower()
    assert "keep calling tools until the task is done" in lowered
    assert "not before each step" in lowered


def test_the_preamble_survives_every_tool_choice():
    """tool_choice changes the call demand, never the anti-narration rule.

    tool_choice='none' forbids calls; the anti-narration instruction must still
    be present, because a 'none' turn that says "I will now check X" is the same
    stall with a different cause.
    """
    for choice in (None, "auto", "none", "required",
                   {"type": "function", "function": {"name": "read_file"}}):
        prompt = _preamble(tool_choice=choice)
        assert "Prose is for an ANSWER" in prompt, f"lost under tool_choice={choice!r}"
        assert "keep calling tools until the task is done" in prompt, (
            f"lost under tool_choice={choice!r}"
        )


def test_the_preamble_survives_a_model_hint():
    """The model hint is appended; it must not displace the instructions."""
    prompt = _preamble(model="us.anthropic.claude-opus-5")
    assert "Prose is for an ANSWER" in prompt
    assert "us.anthropic.claude-opus-5" in prompt


def test_the_preamble_survives_a_tool_list():
    tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }]
    prompt = _preamble(tools=tools)
    assert "Prose is for an ANSWER" in prompt
    assert "read_file" in prompt, "the tool list must still reach the model"


def test_the_tool_call_shape_instruction_is_still_mandatory():
    """The emulated-call contract is what makes any of this work.

    Guarded here because the anti-narration lines sit directly beside it, so an
    edit to one can take the other with it.
    """
    prompt = _preamble()
    assert "<tool_call>" in prompt
    assert "MUST output tool calls" in prompt


@pytest.mark.parametrize("stall_phrase", [
    "If no tool is needed, answer normally",
    "Do what seems best",
])
def test_phrasings_that_permitted_the_stall_never_return(stall_phrase):
    """Pin the two exact strings the mutation harness reverts to.

    These are the mutations that were MISSED before this file existed, so they
    are pinned by value rather than by intent.
    """
    assert stall_phrase not in _preamble()
