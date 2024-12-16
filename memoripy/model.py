from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field


class EmbeddingModel(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.
        """
        pass

    @abstractmethod
    def initialize_embedding_dimension(self) -> int:
        """
        Determine the dimension of the embeddings.
        """
        pass


class ChatModel(ABC):
    @abstractmethod
    def invoke(self, messages: list) -> str:
        """
        Generate a response from the chat model given a list of messages.
        """
        pass

    @abstractmethod
    def extract_concepts(self, text: str) -> list[str]:
        """
        Extract key concepts from the provided text.
        """
        pass

class BaseMessage(BaseModel):
    content: str
    role: str

class HumanMessage(BaseMessage):
    role: Literal["user"] = "user"

class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"

class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"

class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")