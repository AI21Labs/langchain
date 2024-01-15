"""Test AI21LLM llm."""
from langchain_ai21.llms import AI21LLM


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = AI21LLM()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = AI21LLM()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_abatch() -> None:
    """Test streaming tokens from AI21LLM."""
    llm = AI21LLM()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from AI21LLM."""
    llm = AI21LLM()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


def test_batch() -> None:
    """Test batch tokens from AI21LLM."""
    llm = AI21LLM()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from AI21LLM."""
    llm = AI21LLM()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


def test_invoke() -> None:
    """Test invoke tokens from AI21LLM."""
    llm = AI21LLM()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)