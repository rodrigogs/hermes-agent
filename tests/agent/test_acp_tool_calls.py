"""Tool calling over the ACP transport.

This provider has no native function-call channel: the tool schemas are appended
to the prompt text and the reply is parsed back out of it. Everything that makes
an agent loop work therefore lives in how the transcript is rendered, and a defect
there is invisible — the request succeeds, the model answers something plausible,
and only the loop's behaviour is wrong.

What is pinned here is what a second turn needs in order to be interpretable:
   - an assistant turn that made a tool call must survive into the transcript
   - a tool result must say which call it answers
   - tool_choice must be an instruction, not a suggestion
No test spawns a subprocess; the prompt builder and the extractor are the seams.
"""

from __future__ import annotations

import json

from agent import copilot_acp_client as acp


def _transcript(prompt: str) -> str:
    marker = "Conversation transcript:"
    assert marker in prompt, "prompt carries no transcript"
    return prompt[prompt.index(marker):]


WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}


# ---------------------------------------------------------------------------
# the second turn: an assistant tool call must reach the model
# ---------------------------------------------------------------------------

def test_an_assistant_tool_call_is_not_dropped_from_the_transcript() -> None:
    """content=None is how OpenAI represents "I called a tool".

    Rendering only ``content`` made that message empty and skipped it, so the
    model saw a bare ``Tool:`` result with no idea which tool ran, with which
    arguments, or under which id. That is the turn an agent loop depends on.
    """
    messages = [
        {"role": "user", "content": "Weather in Lisbon?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather",
         "content": '{"temp_c": 21}'},
    ]
    body = _transcript(acp._format_messages_as_prompt(messages, tools=[WEATHER_TOOL]))

    assert "Assistant:" in body, "the assistant turn vanished"
    assert "get_weather" in body.split("Tool:")[0], (
        "the call is not in the transcript before its result"
    )
    assert "Lisbon" in body.split("Tool:")[0], "the call's arguments were dropped"


def test_a_tool_result_says_which_call_it_answers() -> None:
    """Two results from two tools arrived as two anonymous blocks, leaving the
    pairing to guesswork."""
    messages = [
        {"role": "user", "content": "Weather and a search."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "web_search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "21C"},
        {"role": "tool", "tool_call_id": "c2", "name": "web_search", "content": "a protocol"},
    ]
    body = _transcript(acp._format_messages_as_prompt(messages, tools=[]))

    weather_block = body.split("21C")[0]
    search_block = body.split("a protocol")[0]
    assert "get_weather" in weather_block.rsplit("Tool:", 1)[-1]
    assert "c1" in weather_block.rsplit("Tool:", 1)[-1]
    assert "web_search" in search_block.rsplit("Tool:", 1)[-1]
    assert "c2" in search_block.rsplit("Tool:", 1)[-1]


def test_an_empty_tool_result_still_appears() -> None:
    """"No output" must read as "completed", not as "never ran".

    An empty result was dropped by the same ``if not rendered: continue`` that
    dropped the assistant turn, so the model saw its own call with no answer at
    all, concluded the tool had not run, and called again. Measured through the
    gateway: a write_file returning "" was retried 12 times until a guardrail
    halted the loop — while the file had been written correctly on the first
    attempt. Writes, deletes and setters succeed silently all the time.
    """
    messages = [
        {"role": "user", "content": "Write X into /tmp/a.txt."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "write_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "write_file", "content": ""},
    ]
    body = _transcript(acp._format_messages_as_prompt(messages, tools=[]))

    assert "Tool:" in body, "the empty result vanished, so the call looks unanswered"
    tool_block = body.rsplit("Tool:", 1)[-1]
    assert "write_file" in tool_block and "c1" in tool_block
    assert "no output" in tool_block, "nothing tells the model the call completed"


def test_a_whitespace_only_tool_result_also_appears() -> None:
    """Same defect, one step subtler: content that renders down to "" after strip."""
    messages = [
        {"role": "user", "content": "Delete /tmp/a.txt."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "delete_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "delete_file", "content": "   \n  "},
    ]
    body = _transcript(acp._format_messages_as_prompt(messages, tools=[]))
    assert "Tool:" in body
    assert "no output" in body.rsplit("Tool:", 1)[-1]


def test_the_sdk_object_shape_round_trips_too() -> None:
    """An agent loop may replay this module's own return value rather than dicts,
    so the renderer has to read attributes as well as keys."""
    call = acp._build_openai_tool_call(
        call_id="c9", name="get_weather", arguments='{"city": "Porto"}',
    )
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
    ]
    body = _transcript(acp._format_messages_as_prompt(messages, tools=[]))
    assert "get_weather" in body
    assert "Porto" in body
    assert "c9" in body


def test_arguments_replayed_as_an_object_are_serialised() -> None:
    """OpenAI sends ``arguments`` as a JSON *string*, but callers replaying history
    frequently hand back the parsed dict. Interpolating a dict would put Python
    repr (single quotes, True/None) into a block the model is told is JSON, so the
    next turn reads malformed protocol.
    """
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": {"city": "Lisbon", "metric": True}}},
        ]},
    ]
    body = _transcript(acp._format_messages_as_prompt(messages, tools=[]))

    block = body[body.index("<tool_call>") + len("<tool_call>"):body.index("</tool_call>")]
    payload = json.loads(block)  # must be valid JSON, not a Python repr
    args = json.loads(payload["function"]["arguments"])
    assert args == {"city": "Lisbon", "metric": True}
    assert "True" not in block, "Python repr leaked into the JSON payload"
    assert "'city'" not in block, "single-quoted keys are not JSON"


def test_assistant_text_and_a_tool_call_both_survive() -> None:
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
        ]},
    ]
    body = _transcript(acp._format_messages_as_prompt(messages, tools=[]))
    assert "Let me check." in body
    assert "get_weather" in body


# ---------------------------------------------------------------------------
# tool_choice is a contract
# ---------------------------------------------------------------------------

def test_a_named_tool_choice_is_an_instruction_not_a_hint() -> None:
    """Measured: appending the raw JSON under the word "hint" let the model answer
    with a friendly greeting and zero calls when a call was required."""
    prompt = acp._format_messages_as_prompt(
        [{"role": "user", "content": "Hello there."}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert "MUST call the tool `get_weather`" in prompt
    assert "hint" not in prompt.lower().split("conversation transcript")[0], (
        "the forcing instruction still reads as optional"
    )


def test_tool_choice_required_demands_some_call() -> None:
    prompt = acp._format_messages_as_prompt(
        [{"role": "user", "content": "Hi."}], tools=[WEATHER_TOOL], tool_choice="required",
    )
    assert "MUST call one of the available tools" in prompt


def test_tool_choice_none_forbids_calls() -> None:
    prompt = acp._format_messages_as_prompt(
        [{"role": "user", "content": "Weather?"}], tools=[WEATHER_TOOL], tool_choice="none",
    )
    assert "DISABLED" in prompt
    assert "emit no" in prompt


def test_tool_choice_auto_stays_optional() -> None:
    prompt = acp._format_messages_as_prompt(
        [{"role": "user", "content": "Weather?"}], tools=[WEATHER_TOOL], tool_choice="auto",
    )
    assert "optional" in prompt
    assert "MUST" not in prompt.split("Conversation transcript")[0].replace(
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls", ""
    )


# ---------------------------------------------------------------------------
# the extractor, on payloads that fight the parser
# ---------------------------------------------------------------------------

def test_arguments_containing_the_delimiter_survive() -> None:
    """A tool argument may legitimately contain the string the parser scans for.

    Verified live: asking the model to write `{"a": "</tool_call>"}` produced a
    correctly escaped call, so the round trip has to preserve it rather than
    truncating at the first delimiter-looking byte.
    """
    payload = {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": "/tmp/x.json", "content": '{"a": "</tool_call>"}'}),
        },
    }
    text = f"Writing it now.\n<tool_call>{json.dumps(payload)}</tool_call>"
    calls, cleaned = acp._extract_tool_calls_from_text(text)

    assert len(calls) == 1
    args = json.loads(calls[0].function.arguments)
    assert args["content"] == '{"a": "</tool_call>"}'
    assert "Writing it now." in cleaned


def test_two_calls_in_one_reply_are_both_extracted() -> None:
    def block(cid: str, city: str) -> str:
        return "<tool_call>" + json.dumps({
            "id": cid, "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
        }) + "</tool_call>"

    calls, _ = acp._extract_tool_calls_from_text(block("c1", "Lisbon") + "\n" + block("c2", "Porto"))
    assert [json.loads(c.function.arguments)["city"] for c in calls] == ["Lisbon", "Porto"]


def test_a_reply_with_no_call_is_left_as_prose() -> None:
    calls, cleaned = acp._extract_tool_calls_from_text("It is 21C and clear in Lisbon.")
    assert calls == []
    assert cleaned == "It is 21C and clear in Lisbon."
