from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from .llm import load_llm
from dotenv import load_dotenv

load_dotenv()

def create_qa_chain(retriever):
    
    llm = load_llm()
    
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
    
    def format_docs(docs):
        """Helper function to format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # To also return the source documents, we can construct a parallel chain
    rag_chain_with_source = RunnableParallel(
        {"answer": rag_chain, "context": retriever}
    )


    print("RAG chain created successfully.")
    return rag_chain_with_source
