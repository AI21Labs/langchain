import pytest
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_ai21 import AI21Embeddings
from langchain_ai21.contextual_answers import (
    AI21ContextualAnswers,
)

context = """
Albert Einstein German: 14 March 1879 – 18 April 1955) 
was a German-born theoretical physicist who is widely held
 to be one of the greatest and most influential scientists 
"""


_GOOD_QUESTION = "When did Albert Einstein born?"
_BAD_QUESTION = "What color is Yoda's light saber?"
_EXPECTED_PARTIAL_RESPONSE = "March 14, 1879"


@pytest.mark.parametrize(
    ids=("good_question", "bad_question"),
    argnames=("question", "expected_answer"),
    argvalues=[
        (_GOOD_QUESTION, _EXPECTED_PARTIAL_RESPONSE),
        (_BAD_QUESTION, "Answer not in context"),
    ],
)
def test_invoke(
    question: str,
    expected_answer: str,
):
    llm = AI21ContextualAnswers()

    response = llm.invoke({"context": context, "question": question})

    assert expected_answer in response


def test_invoke__when_response_if_no_answer_passed__should_use_it():
    response_if_no_answer_found = "This should be the response"
    llm = AI21ContextualAnswers()

    response = llm.invoke(
        input={"context": context, "question": _BAD_QUESTION},
        response_if_no_answer_found=response_if_no_answer_found,
    )

    assert response == response_if_no_answer_found


def test_invoke_when_used_in_a_simple_chain_with_no_vectorstore():
    tsm = AI21ContextualAnswers()

    chain = tsm | StrOutputParser()

    response = chain.invoke(
        {"context": context, "question": _GOOD_QUESTION},
    )

    assert _EXPECTED_PARTIAL_RESPONSE in response


def test_invoke_when_used_in_a_chain_with_vectorstore():
    embeddings = AI21Embeddings()
    faiss = FAISS.from_texts(texts=[context], embedding=embeddings)
    retriever = faiss.as_retriever()

    tsm = AI21ContextualAnswers()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | tsm
        | StrOutputParser()
    )

    response = chain.invoke(_GOOD_QUESTION)

    assert _EXPECTED_PARTIAL_RESPONSE in response
