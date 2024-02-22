from typing import (
    Any,
    Optional,
    Type,
    TypedDict,
)

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from typing_extensions import TypeAlias

from langchain_ai21.ai21_base import AI21Base

ANSWER_NOT_IN_CONTEXT_RESPONSE = "Answer not in context"


class ContextualAnswerInput(TypedDict):
    context: str
    question: str


class AI21ContextualAnswers(RunnableSerializable[ContextualAnswerInput, str], AI21Base):
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def InputType(self) -> TypeAlias:
        """Get the input type for this runnable."""
        return ContextualAnswerInput

    @property
    def OutputType(self) -> Type[str]:
        """Get the input type for this runnable."""
        return str

    def _convert_input(self, input: ContextualAnswerInput) -> ContextualAnswerInput:
        if input.get("context") is None or input.get("question") is None:
            raise ValueError(
                f"Input must contain a 'context' and 'question' field. Got {input}"
            )

        context_input = input["context"]

        if isinstance(context_input, list) and context_input:
            docs = [
                item.page_content if isinstance(item, Document) else item
                for item in context_input
            ]
            return {"context": "\n".join(docs), "question": input["question"]}

        if isinstance(context_input, str):
            return input

        raise ValueError(
            f"Expected input to be a list of strings or Documents."
            f" Received {type(input)}"
        )

    def invoke(
        self,
        input: ContextualAnswerInput,
        config: Optional[RunnableConfig] = None,
        response_if_no_answer_found: str = ANSWER_NOT_IN_CONTEXT_RESPONSE,
        **kwargs: Any,
    ) -> str:
        converted_input = self._convert_input(input)
        response = self.client.answer.create(
            context=converted_input["context"], question=converted_input["question"]
        )

        if response.answer is not None:
            return response.answer

        return response_if_no_answer_found
