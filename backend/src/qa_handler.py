from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from .llm import LLM
from dotenv import load_dotenv

from logger.logger_config import logger

load_dotenv()

class QAHandler:
    """
    A class to handle the creation of the RAG chain.
    """

    def __init__(self, retriever):
        """
        Initializes the QAHandler with a retriever instance.

        Args:
            retriever: The retriever instance to use for the RAG chain.
        """
        self.retriever = retriever
        self.llm = LLM().load()

    def _format_docs(self, docs):
        """
        Helper function to format retrieved documents into a single string.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def create_qa_chain(self):
        """
        Creates and returns the RAG chain.

        Returns:
            The RAG chain instance.
        """
        prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        rag_chain = (
            RunnableParallel(
                {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # To also return the source documents, we can construct a parallel chain
        rag_chain_with_source = RunnableParallel(
            {"answer": rag_chain, "context": self.retriever}
        )

        logger.info("RAG chain created successfully.")
        return rag_chain_with_source
