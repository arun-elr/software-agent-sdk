from typing import cast
from unittest.mock import MagicMock

import pytest

from openhands.sdk.context.condenser.utils import (
    ViewTokenCounter,
    get_shortest_prefix_above_token_count,
    get_suffix_length_for_token_reduction,
    get_total_token_count,
)
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.llm import LLM, Message, TextContent
from openhands.sdk.tool import ToolDefinition
from openhands.sdk.tool.builtins.finish import FinishTool


def message_event(content: str) -> MessageEvent:
    """Helper function to create a MessageEvent for testing."""
    return MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text=content)]),
        source="user",
    )


@pytest.fixture
def mock_llm() -> LLM:
    """Create a mock LLM with token counting capability."""
    mock_llm = MagicMock(spec=LLM)
    mock_llm.model = "test-model"

    # Mock get_token_count to return predictable values based on message content length
    def mock_token_count(messages, **_kwargs):
        # Simple heuristic: count characters in all text content
        # Each character = 0.25 tokens (roughly 4 chars per token)
        total_chars = 0
        for msg in messages:
            for content in msg.content:
                if hasattr(content, "text"):
                    total_chars += len(content.text)
        return total_chars // 4

    mock_llm.get_token_count.side_effect = mock_token_count

    return mock_llm


class TestGetTotalTokenCount:
    """Tests for get_total_token_count function."""

    def test_empty_events(self, mock_llm: LLM):
        """Test with empty event list."""
        events = []
        token_count = get_total_token_count(events, mock_llm)
        assert token_count == 0

    def test_single_event(self, mock_llm: LLM):
        """Test with a single event."""
        events = [message_event("Hello world")]  # 11 chars -> 2 tokens
        token_count = get_total_token_count(events, mock_llm)
        assert token_count == 2

    def test_multiple_events(self, mock_llm: LLM):
        """Test with multiple events."""
        events = [
            message_event("Hello"),  # 5 chars -> 1 token
            message_event("World"),  # 5 chars -> 1 token
            message_event("Test message"),  # 12 chars -> 3 tokens
        ]
        token_count = get_total_token_count(events, mock_llm)
        assert token_count == 5  # (5 + 5 + 12) // 4 = 5

    def test_events_converted_to_messages(self, mock_llm: LLM):
        """Test that events are properly converted to messages."""
        events = [message_event("Test")]
        get_total_token_count(events, mock_llm)

        # Verify get_token_count was called
        assert mock_llm.get_token_count.called  # type: ignore
        # Verify it was called with a list of messages
        call_args = mock_llm.get_token_count.call_args[0][0]  # type: ignore
        assert isinstance(call_args, list)
        assert all(isinstance(msg, Message) for msg in call_args)

    def test_system_prompt_tools_are_counted(self, mock_llm: LLM):
        """Test that agent tool schemas are included in token counting."""
        tools = cast(list[ToolDefinition], list(FinishTool.create()))
        events = [
            SystemPromptEvent(
                source="agent",
                system_prompt=TextContent(text="system"),
                tools=tools,
            ),
            message_event("Test"),
        ]

        mock_llm.get_token_count.side_effect = None  # type: ignore
        mock_llm.get_token_count.return_value = 7  # type: ignore

        assert get_total_token_count(events, mock_llm) == 7

        kwargs = mock_llm.get_token_count.call_args.kwargs  # type: ignore
        assert kwargs["tools"] == tools
        assert kwargs["add_security_risk_prediction"] is True

    def test_system_prompt_empty_tools_are_not_counted(self, mock_llm: LLM):
        """Test that empty tool lists do not add tool schema tokens."""
        events = [
            SystemPromptEvent(
                source="agent",
                system_prompt=TextContent(text="system"),
                tools=[],
            ),
            message_event("Test"),
        ]

        mock_llm.get_token_count.side_effect = None  # type: ignore
        mock_llm.get_token_count.return_value = 3  # type: ignore

        assert get_total_token_count(events, mock_llm) == 3

        kwargs = mock_llm.get_token_count.call_args.kwargs  # type: ignore
        assert kwargs["tools"] is None
        assert kwargs["add_security_risk_prediction"] is False


class TestGetShortestPrefixAboveTokenCount:
    """Tests for get_shortest_prefix_above_token_count function."""

    def test_empty_events(self, mock_llm: LLM):
        """Test with empty event list."""
        events = []
        prefix_length = get_shortest_prefix_above_token_count(events, mock_llm, 10)
        assert prefix_length == 0

    def test_no_prefix_exceeds_token_count(self, mock_llm: LLM):
        """Test when total tokens don't exceed the target."""
        events = [
            message_event("Hi"),  # 2 chars -> 0 tokens
            message_event("Bye"),  # 3 chars -> 0 tokens
        ]
        prefix_length = get_shortest_prefix_above_token_count(events, mock_llm, 100)
        assert prefix_length == len(events)

    def test_single_event_exceeds(self, mock_llm: LLM):
        """Test when first event alone exceeds the token count."""
        events = [
            message_event("A" * 100),  # 100 chars -> 25 tokens
            message_event("B" * 100),  # 100 chars -> 25 tokens
        ]
        prefix_length = get_shortest_prefix_above_token_count(events, mock_llm, 20)
        assert prefix_length == 1

    def test_multiple_events_needed(self, mock_llm: LLM):
        """Test when multiple events are needed to exceed token count."""
        events = [
            message_event("A" * 20),  # 20 chars -> 5 tokens
            message_event("B" * 20),  # 20 chars -> 5 tokens
            message_event("C" * 20),  # 20 chars -> 5 tokens
            message_event("D" * 20),  # 20 chars -> 5 tokens
        ]
        # Need prefix of 3 events to exceed 10 tokens (15 > 10)
        prefix_length = get_shortest_prefix_above_token_count(events, mock_llm, 10)
        assert prefix_length == 3

    def test_exact_boundary(self, mock_llm: LLM):
        """Test behavior at exact token count boundary."""
        events = [
            message_event("A" * 40),  # 40 chars -> 10 tokens
            message_event("B" * 40),  # 40 chars -> 10 tokens
        ]
        # 10 tokens is not > 10, need 2 events for 20 tokens
        prefix_length = get_shortest_prefix_above_token_count(events, mock_llm, 10)
        assert prefix_length == 2

    def test_all_events_needed(self, mock_llm: LLM):
        """Test when all events together just exceed the token count."""
        events = [
            message_event("A" * 16),  # 16 chars -> 4 tokens
            message_event("B" * 16),  # 16 chars -> 4 tokens
            message_event("C" * 16),  # 16 chars -> 4 tokens
        ]
        # Total 12 tokens, need all 3 to exceed 10
        prefix_length = get_shortest_prefix_above_token_count(events, mock_llm, 10)
        assert prefix_length == 3


class TestGetSuffixLengthForTokenReduction:
    """Tests for get_suffix_length_for_token_reduction function."""

    def test_empty_events(self, mock_llm: LLM):
        """Test with empty event list."""
        events = []
        suffix_length = get_suffix_length_for_token_reduction(events, mock_llm, 10)
        assert suffix_length == 0

    def test_zero_token_reduction(self, mock_llm: LLM):
        """Test with zero token reduction requested."""
        events = [
            message_event("Test"),
            message_event("Message"),
        ]
        suffix_length = get_suffix_length_for_token_reduction(events, mock_llm, 0)
        assert suffix_length == len(events)

    def test_negative_token_reduction(self, mock_llm: LLM):
        """Test with negative token reduction (edge case)."""
        events = [
            message_event("Test"),
            message_event("Message"),
        ]
        suffix_length = get_suffix_length_for_token_reduction(events, mock_llm, -10)
        assert suffix_length == len(events)

    def test_small_reduction(self, mock_llm: LLM):
        """Test with small token reduction that removes few events."""
        events = [
            message_event("A" * 40),  # 40 chars -> 10 tokens
            message_event("B" * 40),  # 40 chars -> 10 tokens
            message_event("C" * 40),  # 40 chars -> 10 tokens
            message_event("D" * 40),  # 40 chars -> 10 tokens
        ]
        # Total 40 tokens, reduce by 15 means keep suffix after removing 1 event (10
        # tokens). Actually need to remove 2 events (20 tokens) to exceed 15 token
        # reduction
        suffix_length = get_suffix_length_for_token_reduction(events, mock_llm, 15)
        assert suffix_length == 2  # Keep last 2 events

    def test_large_reduction(self, mock_llm: LLM):
        """Test with large token reduction that removes most events."""
        events = [
            message_event("A" * 20),  # 20 chars -> 5 tokens
            message_event("B" * 20),  # 20 chars -> 5 tokens
            message_event("C" * 20),  # 20 chars -> 5 tokens
            message_event("D" * 20),  # 20 chars -> 5 tokens
        ]
        # Total 20 tokens, reduce by 18 tokens means remove 4 events (20 tokens)
        suffix_length = get_suffix_length_for_token_reduction(events, mock_llm, 18)
        assert suffix_length == 0  # Keep nothing

    def test_exact_reduction(self, mock_llm: LLM):
        """Test with exact token reduction matching some events."""
        events = [
            message_event("A" * 40),  # 40 chars -> 10 tokens
            message_event("B" * 40),  # 40 chars -> 10 tokens
            message_event("C" * 40),  # 40 chars -> 10 tokens
        ]
        # Total 30 tokens, reduce by exactly 10 tokens
        # Need to remove 2 events (20 tokens) to exceed 10 token reduction
        suffix_length = get_suffix_length_for_token_reduction(events, mock_llm, 10)
        assert suffix_length == 1  # Keep last 1 event

    def test_impossible_reduction(self, mock_llm: LLM):
        """Test when requested reduction exceeds total tokens."""
        events = [
            message_event("Hi"),  # 2 chars -> 0 tokens
            message_event("Bye"),  # 3 chars -> 0 tokens
        ]
        # Total ~0 tokens, but asking to reduce by 100
        suffix_length = get_suffix_length_for_token_reduction(events, mock_llm, 100)
        assert suffix_length == 0  # Can't keep anything

    def test_consistency_with_prefix_function(self, mock_llm: LLM):
        """Test that suffix calculation is consistent with prefix calculation."""
        events = [
            message_event("A" * 40),  # 40 chars -> 10 tokens
            message_event("B" * 40),  # 40 chars -> 10 tokens
            message_event("C" * 40),  # 40 chars -> 10 tokens
            message_event("D" * 40),  # 40 chars -> 10 tokens
        ]
        token_reduction = 25

        suffix_length = get_suffix_length_for_token_reduction(
            events, mock_llm, token_reduction
        )
        prefix_length = get_shortest_prefix_above_token_count(
            events, mock_llm, token_reduction
        )

        # Suffix + prefix should equal total length
        assert suffix_length + prefix_length == len(events)

    def test_base_events_are_included_for_prefix_token_counting(self, mock_llm: LLM):
        """Token reduction should count suffixes in the context of kept events."""
        base_events = [
            message_event("system-like prefix"),
            message_event("kept user request"),
        ]
        events = [
            message_event("A" * 40),  # 40 chars -> 10 tokens
            message_event("B" * 40),  # 40 chars -> 10 tokens
            message_event("C" * 40),  # 40 chars -> 10 tokens
        ]

        suffix_length = get_suffix_length_for_token_reduction(
            events,
            mock_llm,
            token_reduction=10,
            base_events=base_events,
        )

        assert suffix_length == 1
        first_counted_messages = mock_llm.get_token_count.call_args_list[0].args[0]  # type: ignore
        counted_text = "\n".join(
            content.text
            for message in first_counted_messages
            for content in message.content
        )
        assert "system-like prefix" in counted_text
        assert "kept user request" in counted_text


def _framed_token_count(messages, **_kwargs) -> int:
    """Per-message framing and per-request priming, as real counters charge."""
    total = 3
    for message in messages:
        total += 3
        for content in message.content:
            total += len(getattr(content, "text", "")) // 4
    return total


class _StubLLM:
    """The parts of ``LLM`` that token counting touches.

    ``ViewTokenCounter`` bounds a view from its parts, and that bound only has
    slack because a standalone count repeats framing a combined count charges
    once, so this charges framing rather than counting characters alone. It
    also keeps ``LLM``'s snapshot and equality behaviour, which decide whether
    a cached count still applies, instead of a mock's identity comparison.
    """

    def __init__(self, scale: int = 1, depends_on_whole_request: bool = False):
        self.scale = scale
        self.token_count_depends_on_whole_request = depends_on_whole_request
        self.calls = 0
        # When set, every count returns this, standing in for a tokenizer that
        # fails and reports 0.
        self.counts: int | None = None

    def get_token_count(self, messages, **kwargs) -> int:
        self.calls += 1
        if self.counts is not None:
            return self.counts
        return self.scale * _framed_token_count(messages, **kwargs)

    def model_copy(self) -> "_StubLLM":
        return _StubLLM(self.scale, self.token_count_depends_on_whole_request)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StubLLM) and (
            self.scale,
            self.token_count_depends_on_whole_request,
        ) == (other.scale, other.token_count_depends_on_whole_request)

    def __hash__(self) -> int:
        return hash((self.scale, self.token_count_depends_on_whole_request))


@pytest.fixture
def framed_llm() -> LLM:
    return cast(LLM, _StubLLM())


def _events(count: int, text: str = "some event text here") -> list:
    return [
        SystemPromptEvent(
            source="agent", system_prompt=TextContent(text="system"), tools=[]
        ),
        *(message_event(f"{i} {text}") for i in range(count)),
    ]


class TestViewTokenCounter:
    """Tests for ViewTokenCounter."""

    def test_skips_the_recount_while_the_view_stays_under_the_cap(
        self, framed_llm: LLM
    ):
        counter = ViewTokenCounter()
        events = _events(4)
        assert counter.count_if_above(events, framed_llm, 10_000) is None
        calls_after_baseline = framed_llm.calls  # type: ignore

        for _ in range(20):
            events.append(message_event("another step"))
            assert counter.count_if_above(events, framed_llm, 10_000) is None

        # Each step counts only the event it appended, never the whole view.
        extra = framed_llm.calls - calls_after_baseline  # type: ignore
        assert extra == 20

    def test_reports_the_exact_count_once_the_view_passes_the_cap(
        self, framed_llm: LLM
    ):
        counter = ViewTokenCounter()
        events = _events(40)

        reported = counter.count_if_above(events, framed_llm, 50)

        assert reported == get_total_token_count(events, framed_llm)
        assert reported is not None and reported > 50

    def test_counts_an_unchanged_view_once(self, framed_llm: LLM):
        counter = ViewTokenCounter()
        events = _events(4)
        # Counted up front, so this assertion's own count is not one of the
        # calls the test then counts.
        expected = get_total_token_count(events, framed_llm)

        first = counter.exact_count(events, framed_llm)
        calls = framed_llm.calls  # type: ignore
        second = counter.exact_count(events, framed_llm)

        assert first == second == expected
        assert framed_llm.calls == calls  # type: ignore

    def test_recounts_when_the_view_no_longer_extends_the_counted_events(
        self, framed_llm: LLM
    ):
        counter = ViewTokenCounter()
        events = _events(10)
        counter.exact_count(events, framed_llm)

        # A condensation drops events from the middle, so the view is no longer
        # an extension of what was counted and nothing can be counted up from.
        condensed = events[:2] + events[6:]

        assert counter.exact_count(condensed, framed_llm) == get_total_token_count(
            condensed, framed_llm
        )

    def test_shorter_view_does_not_reuse_the_longer_counted_total(
        self, framed_llm: LLM
    ):
        counter = ViewTokenCounter()
        events = _events(20)
        counter.exact_count(events, framed_llm)

        truncated = events[:5]

        assert counter.count_if_above(truncated, framed_llm, 1) == (
            get_total_token_count(truncated, framed_llm)
        )

    def test_always_recounts_when_counting_uses_a_chat_template(self, framed_llm: LLM):
        # A chat template renders the whole conversation, so a part of it does
        # not bound the whole and the view must be counted in full every time.
        framed_llm.uses_chat_template_token_count = True  # type: ignore
        counter = ViewTokenCounter()
        events = _events(4)
        counter.count_if_above(events, framed_llm, 10_000)

        calls = framed_llm.calls  # type: ignore
        events.append(message_event("another step"))
        counter.count_if_above(events, framed_llm, 10_000)

        # One more whole-view count, not a single-event count added to a bound.
        assert framed_llm.calls == calls + 1  # type: ignore

    def test_recounts_when_an_appended_event_cannot_be_counted(self, framed_llm: LLM):
        # LLM.get_token_count returns 0 when tokenization fails. Trusting that
        # as the event's size would put the bound under the true count and let
        # the view pass the cap unnoticed, so the view is counted instead.
        counter = ViewTokenCounter()
        events = _events(2)
        assert counter.count_if_above(events, framed_llm, 10_000) is None

        big = message_event("x" * 80_000)
        events.append(big)
        stub = cast(_StubLLM, framed_llm)
        real = stub.get_token_count

        def fail_on_the_new_event_alone(messages, **kwargs):
            if len(messages) == 1 and "x" * 80_000 in str(messages[0].content):
                stub.calls += 1
                return 0
            return real(messages, **kwargs)

        stub.get_token_count = fail_on_the_new_event_alone  # type: ignore[method-assign]

        reported = counter.count_if_above(events, framed_llm, 10_000)

        assert reported is not None and reported > 10_000

    def test_does_not_build_on_a_count_that_failed(self, framed_llm: LLM):
        # LLM.get_token_count returns 0 when tokenization fails, so a whole-view
        # count can come back as 0. Caching that as the baseline would hold the
        # view under every cap for as long as it stayed cached, even after
        # tokenization recovered.
        counter = ViewTokenCounter()
        events = _events(40)
        stub = cast(_StubLLM, framed_llm)
        stub.counts = 0

        assert counter.count_if_above(events, framed_llm, 50) is None

        stub.counts = None
        reported = counter.count_if_above(events, framed_llm, 50)

        assert reported == get_total_token_count(events, framed_llm)

    def test_does_not_reuse_a_count_taken_with_a_different_llm(self, framed_llm: LLM):
        # A count belongs to the tokenizer that produced it.
        counter = ViewTokenCounter()
        events = _events(4)
        counter.exact_count(events, framed_llm)

        verbose = cast(LLM, _StubLLM(scale=10))

        assert counter.exact_count(events, verbose) == get_total_token_count(
            events, verbose
        )

    def test_does_not_track_events_a_condensation_dropped(self, framed_llm: LLM):
        # A condensation that keeps the counted events but replaces what follows
        # leaves a running sum describing events no longer in the view.
        counter = ViewTokenCounter()
        events = _events(2)
        counter.exact_count(events, framed_llm)
        for index in range(30):
            events.append(message_event(f"step {index}"))
            counter.count_if_above(events, framed_llm, 10_000)
            if index % 5 == 4:
                events = events[:3] + events[-2:]
                counter.count_if_above(events, framed_llm, 10_000)

        # Only the events past the counted ones are tracked, so the bookkeeping
        # follows the view's size and not the run's.
        assert len(counter._pending_ids) == len(events) - len(counter._counted_ids)
        assert len(counter._pending_ids) <= len(events)

    def test_does_not_reuse_a_count_after_a_counting_setting_changes(
        self, framed_llm: LLM
    ):
        # An LLM's settings are mutable, so a count is kept against a snapshot
        # of the LLM rather than its identity. Changing a setting that moves
        # token counts drops the count instead of reusing it.
        counter = ViewTokenCounter()
        events = _events(4)
        before = counter.exact_count(events, framed_llm)

        cast(_StubLLM, framed_llm).scale = 10

        after = counter.exact_count(events, framed_llm)

        assert after == get_total_token_count(events, framed_llm)
        assert after != before

    @pytest.mark.parametrize("cap", [1, 40, 120, 400, 10_000])
    def test_decides_exactly_as_a_plain_recount_does(self, framed_llm: LLM, cap: int):
        """The cached path must agree with get_total_token_count at every step."""
        counter = ViewTokenCounter()
        events = _events(0)
        for step in range(30):
            events.append(message_event(f"step {step} " + "body " * (step % 7)))
            if step and step % 11 == 0:
                events = events[:2] + events[5:]
            total = get_total_token_count(events, framed_llm)
            expected = total if total > cap else None
            assert counter.count_if_above(events, framed_llm, cap) == expected
