from collections.abc import Sequence

from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.llm import LLM


def get_total_token_count(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
) -> int:
    """Calculate the total token count for a list of LLM convertible events.

    This function converts the events to LLM messages and uses the provided LLM
    to count the total number of tokens. This is useful for understanding how many
    tokens a sequence of events will consume in the context window. A view is
    expected to have one system prompt event; if multiple are present, only the
    first system prompt's tools are included.

    Args:
        events: List of LLM convertible events to count tokens for
        llm: The LLM instance to use for token counting (uses the litellm's token
            counting utilities)

    Returns:
        Total token count for all events converted to messages

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Hello, how are you?", source="user"),
        ...     MessageEvent.from_text("I'm doing great!", source="agent"),
        ... ]
        >>> token_count = get_total_token_count(events, llm)
        >>> print(f"Total tokens: {token_count}")
    """
    messages = LLMConvertibleEvent.events_to_messages(list(events))
    tools = next(
        (event.tools for event in events if isinstance(event, SystemPromptEvent)),
        None,
    )
    return llm.get_token_count(
        messages,
        tools=tools or None,
        # Security-risk tokens are always included in real tool requests.
        add_security_risk_prediction=bool(tools),
    )


class ViewTokenCounter:
    """Answer "is this view over its token cap?" without recounting it every step.

    :func:`get_total_token_count` re-formats and re-tokenises every event it is
    given, so calling it once per agent step costs time proportional to the view
    on every step, and to the square of the run over the run. A condenser calls
    it to decide one thing: whether the view is over its cap.

    Between condensations a view only grows. An event's standalone count bounds
    what that event adds to a count of the whole view, because the standalone
    count repeats the per-request priming that a combined count charges once. So
    the last exact count, plus the standalone counts of the events past it,
    bounds the view's count from above. While that bound stays at or below the
    cap, the true count does too, and the recount can be skipped.

    A reported count is always exact, so a caller sees the same counts and makes
    the same decisions as one calling :func:`get_total_token_count` directly.

    The bound only holds where a message's count is independent of the rest of
    the request, so it is used only when
    :attr:`LLM.token_count_depends_on_whole_request` is unset.
    """

    def __init__(self) -> None:
        self._llm: LLM | None = None
        self._llm_snapshot: LLM | None = None
        self._counted_ids: tuple[str, ...] = ()
        self._counted_total = 0
        # The events past the counted ones, with the running sum of their
        # standalone counts, so a step costs only the events it added rather
        # than every event since the last exact count.
        self._pending_ids: list[str] = []
        self._pending_sum = 0
        self._pending_bounded = True

    def exact_count(self, events: Sequence[LLMConvertibleEvent], llm: LLM) -> int:
        """Count ``events``, reusing the last count when it still applies.

        Args:
            events: The view's events, in order.
            llm: The LLM whose tokenizer counts the events.

        Returns:
            The exact total token count for ``events``.
        """
        ids = tuple(event.id for event in events)
        if self._has_baseline(llm) and ids == self._counted_ids:
            return self._counted_total
        total = get_total_token_count(events, llm)
        self._rebase(ids, llm, total)
        return total

    def count_if_above(
        self,
        events: Sequence[LLMConvertibleEvent],
        llm: LLM,
        max_tokens: int,
    ) -> int | None:
        """Return the exact count of ``events`` when it is above ``max_tokens``.

        Args:
            events: The view's events, in order.
            llm: The LLM whose tokenizer counts the events.
            max_tokens: The cap to compare the count against.

        Returns:
            The exact total when it is above ``max_tokens``, otherwise ``None``.
            ``None`` is a definite answer and not an unknown one: it is returned
            only once the total is known, or bounded, at or below the cap.
        """
        if not llm.token_count_depends_on_whole_request and self._extend(events, llm):
            if self._pending_bounded:
                if self._counted_total + self._pending_sum <= max_tokens:
                    return None
        total = self.exact_count(events, llm)
        return total if total > max_tokens else None

    def _has_baseline(self, llm: LLM) -> bool:
        """Whether there is a count to build on, valid for how ``llm`` counts."""
        # A count belongs to the settings that produced it, and an LLM's
        # settings are mutable, so the count is kept against a snapshot of the
        # LLM rather than its identity: changing a setting that moves token
        # counts, such as vision or the tokenizer, drops the count. Comparing
        # every field means a setting added later drops it too, rather than
        # being silently assumed not to matter. The snapshot is shallow, so a
        # setting mutated in place inside a nested object is not seen.
        #
        # A count of 0 is what LLM.get_token_count reports when tokenization
        # failed. Trusting it would hold the view under every cap for as long
        # as it stayed cached, so it is not built on either.
        if self._llm_snapshot is None or self._counted_total <= 0:
            return False
        return self._llm_snapshot == llm

    def _rebase(self, ids: tuple[str, ...], llm: LLM, total: int) -> None:
        self._llm = llm
        self._llm_snapshot = llm.model_copy()
        self._counted_ids = ids
        self._counted_total = total
        self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending_ids = []
        self._pending_sum = 0
        self._pending_bounded = True

    def _extend(self, events: Sequence[LLMConvertibleEvent], llm: LLM) -> bool:
        """Bring the running sum up to date, or report no usable baseline.

        The baseline is usable for any view that starts with the events it
        counted, which an appended view does and a view that condensed inside
        that prefix does not.
        """
        if not self._has_baseline(llm):
            return False
        counted = len(self._counted_ids)
        if len(events) < counted:
            return False
        if tuple(event.id for event in events[:counted]) != self._counted_ids:
            return False

        past_baseline = list(events[counted:])
        ids = [event.id for event in past_baseline]
        summed = len(self._pending_ids)
        if ids[:summed] != self._pending_ids:
            # The events past the baseline changed, as after a condensation
            # that kept the counted ones, so the running sum no longer
            # describes them and is rebuilt from what is there now.
            self._clear_pending()
            summed = 0
        for event in past_baseline[summed:]:
            total = get_total_token_count([event], llm)
            if total <= 0:
                # Tokenization failed for this event, so its contribution is
                # unknown and the sum no longer bounds anything.
                self._pending_bounded = False
            self._pending_sum += total
            self._pending_ids.append(event.id)
        return True


def get_shortest_prefix_above_token_count(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
    token_count: int,
    base_events: Sequence[LLMConvertibleEvent] | None = None,
) -> int:
    """Find the length of the shortest prefix whose token count exceeds the target.

    This function performs a binary search to efficiently find the shortest prefix
    of events that, when converted to messages, has a total token count greater than
    the specified target token count.

    Args:
        events: List of LLM convertible events to search through
        llm: The LLM instance to use for token counting (uses the model's tokenizer)
        token_count: The target token count threshold

    Returns:
        The length of the shortest prefix that exceeds the token count.
        Returns 0 if no events are provided.
        Returns len(events) if all events combined don't exceed the token count.

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Hi", source="user"),
        ...     MessageEvent.from_text("Hello", source="agent"),
        ...     MessageEvent.from_text("How are you?", source="user"),
        ...     MessageEvent.from_text("Great!", source="agent"),
        ... ]
        >>> prefix_len = get_shortest_prefix_above_token_count(events, llm, 20)
        >>> # prefix_len might be 2 if first 2 events exceed 20 tokens
    """
    if not events:
        return 0

    base_events = base_events or []
    base_tokens = get_total_token_count(base_events, llm) if base_events else 0

    # Check if all events combined don't exceed the token count
    total_tokens = get_total_token_count([*base_events, *events], llm) - base_tokens
    if total_tokens <= token_count:
        return len(events)

    # Binary search for the shortest prefix
    left, right = 1, len(events)

    while left < right:
        mid = (left + right) // 2
        prefix_tokens = (
            get_total_token_count([*base_events, *events[:mid]], llm) - base_tokens
        )

        if prefix_tokens > token_count:
            # This prefix exceeds the count, try to find a shorter one
            right = mid
        else:
            # This prefix doesn't exceed, we need a longer one
            left = mid + 1

    return left


def get_suffix_length_for_token_reduction(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
    token_reduction: int,
    base_events: Sequence[LLMConvertibleEvent] | None = None,
) -> int:
    """Find how many suffix events can be kept while reducing tokens by target amount.

    This function determines the maximum number of events from the end of the list
    that can be retained while ensuring the total token count is reduced by at least
    the specified amount. It uses the get_shortest_prefix_above_token_count function
    to find the prefix that must be removed.

    Args:
        events: List of LLM convertible events
        llm: The LLM instance to use for token counting (uses the model's tokenizer)
        token_reduction: The minimum number of tokens to reduce by

    Returns:
        The number of events from the end that can be kept (suffix length).

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Event 1", source="user"),
        ...     MessageEvent.from_text("Event 2", source="agent"),
        ...     MessageEvent.from_text("Event 3", source="user"),
        ...     MessageEvent.from_text("Event 4", source="agent"),
        ... ]
        >>> # Suppose total is 100 tokens, and we want to reduce by 40 tokens
        >>> suffix_len = get_suffix_length_for_token_reduction(events, llm, 40)
        >>> # suffix_len tells us how many events from the end we can keep
        >>> # If first 2 events = 45 tokens, suffix_len = 2 (keep last 2 events)
    """
    if not events:
        return 0

    if token_reduction <= 0:
        return len(events)

    # Find the shortest prefix that exceeds the token reduction target
    prefix_length = get_shortest_prefix_above_token_count(
        events,
        llm,
        token_reduction,
        base_events=base_events,
    )

    # The suffix length is what remains after removing the prefix
    suffix_length = len(events) - prefix_length

    return suffix_length
