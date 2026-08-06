"""The requested model must reach the ACP subprocess environment.

The ACP wire protocol carries no model field and the agent process reads its
model from the environment, so HERMES_ACP_MODEL is the only channel between
"the user picked model X" and the process that has to serve it. Without these
tests a refactor can drop the plumbing and every session silently falls back to
whatever default the far side pinned — a failure that looks like success.
"""

from agent.copilot_acp_client import _build_subprocess_env


class TestSubprocessModelEnv:
    def test_model_is_exported(self):
        env = _build_subprocess_env("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        assert env["HERMES_ACP_MODEL"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def test_absent_when_no_model_requested(self):
        # The far side must keep its own default rather than receive an empty
        # string, which a naive bridge could forward as a real (invalid) id.
        assert "HERMES_ACP_MODEL" not in _build_subprocess_env()
        assert "HERMES_ACP_MODEL" not in _build_subprocess_env(None)

    def test_empty_and_falsy_models_are_not_exported(self):
        assert "HERMES_ACP_MODEL" not in _build_subprocess_env("")

    def test_credentials_still_inherited(self):
        # Regression guard: the model argument must not disturb the existing
        # credential-inheriting behaviour the ACP executor depends on.
        env = _build_subprocess_env("us.anthropic.claude-opus-5")
        assert env.get("HOME")


class TestRunPromptForwardsModel:
    def test_run_prompt_accepts_model_kwarg(self):
        import inspect

        from agent.copilot_acp_client import CopilotACPClient

        sig = inspect.signature(CopilotACPClient._run_prompt)
        assert "model" in sig.parameters
        # Keyword-only with a default, so existing positional callers still work.
        assert sig.parameters["model"].default is None

    def test_completion_path_passes_model_through(self, monkeypatch):
        """The model given to _create_chat_completion must reach _run_prompt."""
        from agent.copilot_acp_client import CopilotACPClient

        client = CopilotACPClient.__new__(CopilotACPClient)
        seen = {}

        def fake_run_prompt(prompt_text, *, timeout_seconds, model=None):
            seen["model"] = model
            return ("done", "")

        monkeypatch.setattr(client, "_run_prompt", fake_run_prompt)
        client._create_chat_completion(
            model="us.anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert seen["model"] == "us.anthropic.claude-sonnet-5"
