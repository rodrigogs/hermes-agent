"""The ``$HERMES_HOME/hooks/worktree-created`` seam on kanban worktree creation.

A linked worktree is a fresh checkout: it holds tracked files and nothing else,
so every gitignored file is absent by definition — including the ``.env`` most
projects cannot start without. Which secrets a project needs is deployment
knowledge no code in this repo can hold, so ``_ensure_git_worktree`` hands the
operator a seam right after ``git worktree add`` succeeds.

These tests drive a real executable hook under a real temp ``HERMES_HOME``
against a real ``git worktree add`` — the argv order, the cwd, and the failure
modes are the contract external hooks are written against, so none of them may
drift silently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME (no kanban DB needed — this is worktree plumbing)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A one-commit git repo to hang linked worktrees off."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    (root / "README.md").write_text("base\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "init")
    return root


def _install_hook(home: Path, body: str, *, executable: bool = True) -> Path:
    hook = home / "hooks" / "worktree-created"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    hook.chmod(0o755 if executable else 0o644)
    return hook


# ---------------------------------------------------------------------------
# The hook runs, with the documented argv and cwd
# ---------------------------------------------------------------------------


def test_hook_gets_worktree_repo_branch_and_runs_inside_the_worktree(
    kanban_home: Path, repo: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """argv is ``<worktree> <repo-root> <branch>``, cwd is the new worktree."""
    log = tmp_path / "hook-argv.log"
    _install_hook(
        kanban_home,
        f'printf "%s|%s|%s|%s\\n" "$1" "$2" "$3" "$(pwd -P)" >> "{log}"\n'
        'echo "seeded 2 files"\n',
    )
    target = repo / ".worktrees" / "t_abc"

    with caplog.at_level("INFO", logger=kb._log.name):
        kb._ensure_git_worktree(repo, target, "wt/t_abc")

    assert target.is_dir()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, f"hook ran {len(lines)} times, expected exactly once"
    worktree_arg, repo_arg, branch_arg, cwd_arg = lines[0].split("|")
    assert Path(worktree_arg) == target
    assert Path(repo_arg) == repo
    assert branch_arg == "wt/t_abc"
    assert Path(cwd_arg).resolve() == target.resolve()
    # Hook stdout is surfaced, otherwise a hook that reports what it seeded
    # reports it into a void.
    assert "seeded 2 files" in caplog.text


def test_existing_worktree_does_not_re_run_the_hook(
    kanban_home: Path, repo: Path, tmp_path: Path,
) -> None:
    """The hook is a *created* event: the idempotent no-op path stays silent."""
    log = tmp_path / "hook-argv.log"
    _install_hook(kanban_home, f'echo ran >> "{log}"\n')
    target = repo / ".worktrees" / "t_once"

    kb._ensure_git_worktree(repo, target, "wt/t_once")
    kb._ensure_git_worktree(repo, target, "wt/t_once")

    assert log.read_text(encoding="utf-8").splitlines() == ["ran"]


# ---------------------------------------------------------------------------
# No hook installed / not executable — nothing happens
# ---------------------------------------------------------------------------


def test_no_hook_installed_is_not_an_error(kanban_home: Path, repo: Path) -> None:
    target = repo / ".worktrees" / "t_nohook"

    kb._ensure_git_worktree(repo, target, "wt/t_nohook")

    assert target.is_dir()
    assert not (kanban_home / "hooks").exists()


def test_non_executable_hook_is_ignored(
    kanban_home: Path, repo: Path, tmp_path: Path,
) -> None:
    """Only ``chmod +x`` is consent: a mode-644 file is a draft, not a hook."""
    log = tmp_path / "hook-argv.log"
    hook = _install_hook(kanban_home, f'echo ran >> "{log}"\n', executable=False)
    target = repo / ".worktrees" / "t_noexec"

    kb._ensure_git_worktree(repo, target, "wt/t_noexec")

    assert target.is_dir()
    assert not os.access(hook, os.X_OK)
    assert not log.exists()


# ---------------------------------------------------------------------------
# Failure modes are loud
# ---------------------------------------------------------------------------


def test_failing_hook_raises_with_its_stderr(kanban_home: Path, repo: Path) -> None:
    """Exit != 0 raises, and the hook's own stderr is in the message."""
    _install_hook(kanban_home, 'echo "no .env for this project" >&2\nexit 1\n')
    target = repo / ".worktrees" / "t_fail"

    with pytest.raises(RuntimeError) as excinfo:
        kb._ensure_git_worktree(repo, target, "wt/t_fail")

    message = str(excinfo.value)
    assert "worktree-created" in message
    assert "no .env for this project" in message
    # The worktree itself was created before the hook ran; the raise is about
    # the seeding, not about git, and the message must not blame git.
    assert "git worktree add failed" not in message
    assert target.is_dir()


def test_hook_timeout_raises(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hook that hangs fails the worktree instead of wedging the dispatcher."""
    # ``exec >/dev/null 2>&1`` hands the pipes back before sleeping, so killing
    # the shell on timeout is not held up waiting for ``sleep`` to exit.
    _install_hook(kanban_home, "exec >/dev/null 2>&1\nsleep 30\n")
    monkeypatch.setattr(kb, "_WORKTREE_CREATED_HOOK_TIMEOUT_SECONDS", 0.5)
    target = repo / ".worktrees" / "t_hang"

    with pytest.raises(RuntimeError, match="worktree-created"):
        kb._ensure_git_worktree(repo, target, "wt/t_hang")
