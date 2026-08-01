"""Tests for hermes_cli/tips.py — random tip display at session start."""

from hermes_cli.tips import TIPS, get_random_tip


class TestTipsCorpus:
    """Validate the tip corpus itself."""

    def test_has_at_least_200_tips(self):
        assert len(TIPS) >= 200, f"Expected 200+ tips, got {len(TIPS)}"

    def test_no_tip_promises_that_data_stays_local(self):
        """A tip is documentation, and a false security claim is worse than none.

        One tip read "zero data leaves localhost" about `hermes dashboard`. Both
        halves are wrong: the dashboard talks to whichever LLM provider is
        configured, so data leaves the machine by design, and the product itself
        documents `hermes dashboard --host 0.0.0.0`, which makes the bind reachable
        from the network. An operator deciding whether to expose the dashboard is
        exactly who reads these tips.

        Describing the default bind is fine. Promising confinement is not.
        """
        forbidden = (
            "zero data leaves",
            "no data leaves",
            "data never leaves",
            "nothing leaves your machine",
            "stays on your machine",
            "100% local",
            "fully offline",
        )
        for i, tip in enumerate(TIPS):
            low = tip.lower()
            for phrase in forbidden:
                assert phrase not in low, (
                    f"Tip {i} promises data confinement the product does not "
                    f"guarantee: {tip!r}"
                )

    def test_no_duplicates(self):
        assert len(TIPS) == len(set(TIPS)), "Duplicate tips found"

    def test_all_tips_are_strings(self):
        for i, tip in enumerate(TIPS):
            assert isinstance(tip, str), f"Tip {i} is not a string: {type(tip)}"

    def test_no_empty_tips(self):
        for i, tip in enumerate(TIPS):
            assert tip.strip(), f"Tip {i} is empty or whitespace-only"

    def test_max_length_reasonable(self):
        """Tips should fit on a single terminal line (~120 chars max)."""
        for i, tip in enumerate(TIPS):
            assert len(tip) <= 150, (
                f"Tip {i} too long ({len(tip)} chars): {tip[:60]}..."
            )

    def test_no_leading_trailing_whitespace(self):
        for i, tip in enumerate(TIPS):
            assert tip == tip.strip(), f"Tip {i} has leading/trailing whitespace"


class TestGetRandomTip:
    """Validate the get_random_tip() function."""

    def test_returns_string(self):
        tip = get_random_tip()
        assert isinstance(tip, str)
        assert len(tip) > 0

    def test_returns_tip_from_corpus(self):
        tip = get_random_tip()
        assert tip in TIPS

    def test_randomness(self):
        """Multiple calls should eventually return different tips."""
        seen = set()
        for _ in range(50):
            seen.add(get_random_tip())
        # With 200+ tips and 50 draws, we should see at least 10 unique
        assert len(seen) >= 10, f"Only got {len(seen)} unique tips in 50 draws"


class TestTipIntegrationInCLI:
    """Test that the tip display code in cli.py works correctly."""

    def test_tip_import_works(self):
        """The import used in cli.py must succeed."""
        from hermes_cli.tips import get_random_tip
        assert callable(get_random_tip)

    def test_tip_display_format(self):
        """Verify the Rich markup format doesn't break."""
        tip = get_random_tip()
        color = "#B8860B"
        markup = f"[dim {color}]✦ Tip: {tip}[/]"
        # Should not contain nested/broken Rich tags
        assert markup.count("[/]") == 1
        assert "[dim #B8860B]" in markup
