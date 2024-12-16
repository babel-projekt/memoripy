import json
from typing import List
import numpy as np
import ollama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from prompts import CONCEPTS_EXTRACTION
from model import ChatModel, EmbeddingModel, ConceptExtractionResponse, HumanMessage, SystemMessage, AssistantMessage, BaseMessage
from openai import OpenAI


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key, model_name="text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name
        self.embeddings_model = OpenAIEmbeddings(model=model_name, api_key=self.api_key)

        if model_name == "text-embedding-3-small":
            self.dimension = 1536
        else:
            raise ValueError("Unsupported OpenAI embedding model name for specified dimension.")

    def get_embedding(self, text: str) -> np.ndarray:
        embedding = self.embeddings_model.embed_query(text)
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        return np.array(embedding)

    def initialize_embedding_dimension(self) -> int:
        return self.dimension


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name="mxbai-embed-large"):
        self.model_name = model_name
        self.dimension = self.initialize_embedding_dimension()

    def get_embedding(self, text: str) -> np.ndarray:
        response = ollama.embeddings(model=self.model_name, prompt=text)
        embedding = response.get("embedding")
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        return np.array(embedding)

    def initialize_embedding_dimension(self) -> int:
        test_text = "Test to determine embedding dimension"
        response = ollama.embeddings(
            model=self.model_name,
            prompt=test_text
        )
        embedding = response.get("embedding")
        if embedding is None:
            raise ValueError("Failed to retrieve embedding for dimension initialization.")
        return len(embedding)


class OpenAIChatModel(ChatModel):
    def __init__(self, api_key, model_name="gpt-3.5-turbo", base_url=None):
        self.api_key = api_key
        self.model_name = model_name
        if base_url:
            self.llm = ChatOpenAI(model=model_name, api_key=self.api_key, base_url=base_url)
        else:
            self.llm = ChatOpenAI(model=model_name, api_key=self.api_key)
        self.parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)
        self.prompt_template = PromptTemplate(
            template=(
                "Extract key concepts from the following text in a concise, context-specific manner. "
                "Include only highly relevant and specific concepts.\n"
                "{format_instructions}\n{text}"
            ),
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        print(self.prompt_template)

    def invoke(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        return str(response.content)

    def extract_concepts(self, text: str) -> list[str]:
        chain = self.prompt_template | self.llm | self.parser
        response = chain.invoke({"text": text})
        concepts = response.get("concepts", [])
        print(f"Concepts extracted: {concepts}")
        return concepts


class OllamaChatModel(ChatModel):
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)
        self.prompt_template = PromptTemplate(
            template=(
                "Please analyze the following text and provide a list of key concepts that are unique to this content. "
                "Return only the core concepts that best capture the text's meaning.\n"
                "{format_instructions}\n{text}"
            ),
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def invoke(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        return str(response.content)

    def extract_concepts(self, text: str) -> list[str]:
        chain = self.prompt_template | self.llm | self.parser
        response = chain.invoke({"text": text})
        concepts = response.get("concepts", [])
        print(f"Concepts extracted: {concepts}")
        return concepts


class ChatModelV2(ChatModel):

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url
        )

    def converted_messages(
        self,
        messages: List[BaseMessage]
    ):
        return [
            message.model_dump(exclude_none=True)
            for message in messages
        ]

    def invoke(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.converted_messages(messages),
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def extract_concepts(self, text: str) -> list[str]:
        retries = 2
        messages = [
            SystemMessage(content=CONCEPTS_EXTRACTION.format(text=text)),
            HumanMessage(content="Now extract the concepts from the text.")
        ]
        try:
            bare_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.converted_messages(messages),
                max_tokens=1024,
                temperature=0,
            )
            bare_response = bare_response.choices[0].message.content
            model_response = ConceptExtractionResponse.model_validate_json(bare_response)
            return model_response.concepts
        except Exception as e:
            if retries > 0:
                messages.extend(
                    [
                        AssistantMessage(content=bare_response),
                        HumanMessage(content="Error in generated JSON response: " + str(e) + ". Please try again.")
                    ]
                )
                retries -= 1
                return self.extract_concepts(text, retries, messages)
            else:
                return []
