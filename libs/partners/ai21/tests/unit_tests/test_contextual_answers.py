from typing import Dict

import pytest
from langchain_core.documents import Document

from langchain_ai21 import AI21ContextualAnswers


@pytest.mark.parametrize(
    ids=[
        "when_no_context__should_raise_exception",
        "when_no_question__should_raise_exception",
    ],
    argnames="input",
    argvalues=[
        ({"question": "What is the capital of France?"}),
        ({"context": "Paris is the capital of France"}),
    ],
)
def test_invoke__on_bad_input(input: Dict[str, str]):
    tsm = AI21ContextualAnswers()

    with pytest.raises(ValueError) as error:
        tsm.invoke(input)

    assert (
        error.value.args[0]
        == f"Input must contain a 'context' and 'question' field. Got {input}"
    )


@pytest.mark.parametrize(
    ids=[
        "when_context_is_an_empty_list",
        "when_context_is_not_str_or_list_of_docs_or_str",
    ],
    argnames="input",
    argvalues=[
        ({"context": [], "question": "What is the capital of France?"}),
        ({"context": 1242, "question": "What is the capital of France?"}),
    ],
)
def test_invoke__on_bad_input(input: Dict[str, str]):
    tsm = AI21ContextualAnswers()

    with pytest.raises(ValueError) as error:
        tsm.invoke(input)

    assert (
        error.value.args[0]
        == f"Expected input to be a list of strings or Documents. Received {type(input)}"
    )


@pytest.mark.parametrize(
    ids=[
        "when_context_is_a_list_of_strings",
        "when_context_is_a_list_of_documents",
        "when_context_is_a_string",
    ],
    argnames="input",
    argvalues=[
        (
            {
                "context": ["Paris is the capital of france"],
                "question": "What is the capital of France?",
            }
        ),
        (
            {
                "context": [Document(page_content="Paris is the capital of france")],
                "question": "What is the capital of France?",
            }
        ),
        (
            {
                "context": "Paris is the capital of france",
                "question": "What is the capital of France?",
            }
        ),
    ],
)
def test_invoke__on_good_input(input: Dict[str, str]):
    tsm = AI21ContextualAnswers()

    response = tsm.invoke(input)
    assert isinstance(response, str)
