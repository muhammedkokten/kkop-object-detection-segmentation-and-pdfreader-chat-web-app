from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os

os.environ["TOGETHER_API_KEY"] = "USE API KEY IN HERE"

import together

together.api_key = os.environ["TOGETHER_API_KEY"]
models = together.Models.list()

together.Models.start("togethercomputer/llama-2-70b-chat")

import together

import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

def chat_ai(user_message):
    class TogetherLLM(LLM):
        """Together large language models."""

        model: str = "togethercomputer/llama-2-70b-chat"
        """model endpoint to use"""

        together_api_key: str = os.environ["TOGETHER_API_KEY"]
        """Together API key"""

        temperature: float = 0.7
        """What sampling temperature to use."""

        max_tokens: int = 512
        """The maximum number of tokens to generate in the completion."""

        class Config:
            extra = Extra.forbid

        @root_validator()
        def validate_environment(cls, values: Dict) -> Dict:
            """Validate that the API key is set."""
            api_key = get_from_dict_or_env(
                values, "together_api_key", "TOGETHER_API_KEY"
            )
            values["together_api_key"] = api_key
            return values

        @property
        def _llm_type(self) -> str:
            """Return type of LLM."""
            return "together"

        def _call(
            self,
            prompt: str,
            **kwargs: Any,
        ) -> str:
            """Call to Together endpoint."""
            together.api_key = self.together_api_key
            output = together.Complete.create(prompt,
                                              model=self.model,
                                              max_tokens=self.max_tokens,
                                              temperature=self.temperature,
                                              )
            text = output['output']['choices'][0]['text']
            return text

    loader = DirectoryLoader('uploads/pdf', glob="./*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    from langchain.embeddings import HuggingFaceInstructEmbeddings

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                          model_kwargs={"device": "cuda"})

    persist_directory = 'db'
    embedding=instructor_embeddings
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = TogetherLLM(
        model= "togethercomputer/llama-2-70b-chat",
        temperature = 0.1,
        max_tokens = 1024
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                      chain_type="stuff",
                                      retriever=retriever,
                                      return_source_documents=True)

    import textwrap

    def wrap_text_preserve_newlines(text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text

    def process_llm_response(llm_response):
        print(wrap_text_preserve_newlines(llm_response['result']))
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])

    query = user_message
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    return llm_response["result"]
