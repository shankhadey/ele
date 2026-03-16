"""Shared Hypothesis generators for the AI Model Evaluation Workflow tests."""

from hypothesis import strategies as st

from evaluation_workflow.core.models import (
    CategoryEnum,
    DomainEnum,
    DifficultyEnum,
    AnswerFormatEnum,
    StatusEnum,
    RunStatusEnum,
    ResultStatusEnum,
    Contributor,
    Scenario,
)


# --- Enum strategies ---

categories = st.sampled_from(list(CategoryEnum))
domains = st.sampled_from(list(DomainEnum))
difficulties = st.sampled_from(list(DifficultyEnum))
answer_formats = st.sampled_from(list(AnswerFormatEnum))
statuses = st.sampled_from(list(StatusEnum))
run_statuses = st.sampled_from(list(RunStatusEnum))
result_statuses = st.sampled_from(list(ResultStatusEnum))


# --- Helper: generate text with a word count in a given range ---

def _text_with_word_count(min_words: int, max_words: int) -> st.SearchStrategy[str]:
    """Generate a string whose word count is between min_words and max_words."""
    return st.integers(min_value=min_words, max_value=max_words).flatmap(
        lambda n: st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L",), min_codepoint=97, max_codepoint=122),
                min_size=3,
                max_size=10,
            ),
            min_size=n,
            max_size=n,
        ).map(lambda words: " ".join(words))
    )


# --- Contributor strategy ---

def _non_blank_text(min_size: int = 1, max_size: int = 50) -> st.SearchStrategy[str]:
    """Generate text that contains at least one non-whitespace character."""
    return st.text(
        min_size=min_size,
        max_size=max_size,
        alphabet=st.characters(whitelist_categories=("L", "Zs")),
    ).filter(lambda t: t.strip())


@st.composite
def contributors(draw: st.DrawFn) -> Contributor:
    return Contributor(
        name=draw(_non_blank_text(1, 50)),
        title=draw(_non_blank_text(1, 50)),
        organization=draw(_non_blank_text(1, 50)),
        years_experience=draw(st.integers(min_value=0, max_value=50)),
        domain_expertise=draw(_non_blank_text(1, 100)),
    )


# --- Scenario strategy ---

@st.composite
def valid_scenarios(draw: st.DrawFn) -> Scenario:
    """Generate a fully valid Scenario instance."""
    fmt = draw(answer_formats)
    choices = []
    correct_answer = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L",))))

    if fmt == AnswerFormatEnum.MULTIPLE_CHOICE:
        num_choices = draw(st.integers(min_value=2, max_value=6))
        # Build a choices list that includes the correct answer
        other_choices = draw(
            st.lists(
                st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L",))),
                min_size=num_choices - 1,
                max_size=num_choices - 1,
            )
        )
        insert_pos = draw(st.integers(min_value=0, max_value=len(other_choices)))
        choices = list(other_choices)
        choices.insert(insert_pos, correct_answer)

    return Scenario(
        title=draw(_non_blank_text(1, 100)),
        category=draw(categories),
        domain=draw(domains),
        difficulty=draw(difficulties),
        scenario_text=draw(_text_with_word_count(200, 500)),
        question=draw(_non_blank_text(1, 200)),
        answer_format=fmt,
        correct_answer=correct_answer,
        rationale=draw(_text_with_word_count(100, 300)),
        contributor=draw(contributors()),
        choices=choices,
        tools_available=draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        status=draw(statuses),
    )
