from abc import ABC
from typing import Optional, Union, Any, Type, Tuple, Sequence, TypeAlias, List

from langchain_ai21.ai21_base import AI21Base
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models.base import (
    LanguageModelOutput,
    BaseLanguageModel,
)
from langchain_core.load import Serializable
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import StringPromptValue, PromptValue
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.utils import Input, Output

TSMModelInput = Union[StringPromptValue, str]


class ContextualAnswerBody(Serializable):
    context: str
    question: str


ContextualAnswerStrDocType = Tuple[str, Sequence[Document]]
ContextualAnswerInputType = Union[
    ContextualAnswerBody,
    ContextualAnswerStrDocType,
]


class AI21ContextualAnswersLLM(
    RunnableSerializable[ContextualAnswerInputType, str], AI21Base
):
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def InputType(self) -> TypeAlias:
        """Get the input type for this runnable."""
        return ContextualAnswerInputType

    @property
    def OutputType(self) -> Type[str]:
        """Get the input type for this runnable."""
        return str

    def _convert_input(
        self, input: ContextualAnswerInputType
    ) -> ContextualAnswerBody:
        if isinstance(input, ContextualAnswerBody):
            return input

        if isinstance(input, tuple):
            if isinstance(input[1], Document):
                docs = "\n".join([document.page_content for document in input[1]])
                return ContextualAnswerBody(context=docs, question=input[0])

        raise ValueError(f"Invalid input type {type(input)}")

    def invoke(
        self,
        input: ContextualAnswerInputType,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> LanguageModelOutput:
        converted_input = self._convert_input(input)
        response = self.client.answer.create(
            context=converted_input.context, question=converted_input.question
        )

        if response.answer_in_context:
            return response.answer

        return "Answer Not in Context"


class AI21ContextualAnswer(BaseLanguageModel[ContextualAnswerInputType], AI21Base):
    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        pass

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        pass

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        pass

    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        raise DeprecationWarning("Use `generate_prompt` instead")

    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        raise DeprecationWarning("Use `generate_prompt` instead")

    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        raise DeprecationWarning("Use `generate_prompt` instead")

    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        raise DeprecationWarning("Use `generate_prompt` instead")
