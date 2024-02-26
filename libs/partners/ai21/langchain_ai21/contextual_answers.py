from typing import (
    Any,
    Optional,
    Type,
    TypedDict,
)

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
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
        context = input.get("context")
        question = input.get("question")

        if not context or not question:
            raise ValueError(
                f"Input must contain a 'context' and 'question' fields. Got {input}"
            )

        if isinstance(context, list):
            docs = [
                item.page_content if isinstance(item, Document) else item
                for item in context
            ]
            return {"context": "\n".join(docs), "question": input["question"]}

        if isinstance(context, str):
            return input

        raise ValueError(
            f"Expected input to be a list of strings or Documents."
            f" Received {type(input)}"
        )

    def _call_contextual_answers_model(self, input: ContextualAnswerInput) -> str:
        converted_input = self._convert_input(input)
        return self.client.answer.create(
            context=converted_input["context"], question=converted_input["question"]
        ).answer

    def invoke(
        self,
        input: ContextualAnswerInput,
        config: Optional[RunnableConfig] = None,
        response_if_no_answer_found: str = ANSWER_NOT_IN_CONTEXT_RESPONSE,
        **kwargs: Any,
    ) -> str:
        config = ensure_config(config)
        answer = self._call_with_config(
            func=self._call_contextual_answers_model,
            input=input,
            config=config,
            run_type="llm",
        )

        if answer is None:
            return response_if_no_answer_found

        return answer
