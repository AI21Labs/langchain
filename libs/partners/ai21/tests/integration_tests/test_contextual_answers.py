import pytest

from langchain_ai21 import AI21Embeddings
from langchain_ai21.contextual_answers import (
    AI21ContextualAnswers,
)
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

context = """
Albert Einstein (/ˈaɪnstaɪn/ EYEN-styne;[4] German: [ˈalbɛɐt ˈʔaɪnʃtaɪn] ⓘ; 14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century.[1][5] His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation".[6] He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect",[7] a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science.[8][9] In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time.[10] His intellectual achievements and originality have made the word Einstein broadly synonymous with genius.[11]
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


def test_invoke__when_response_if_no_answer_passed__should_use_response_if_no_answer_passed():
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
