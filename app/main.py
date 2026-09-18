import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


load_dotenv()


class AgentState(TypedDict):
    question: str
    context: str
    answer: str


documents = [
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) combines information retrieval "
            "with large language models. A retriever searches a knowledge base "
            "for relevant information, and the language model uses that context "
            "to generate a grounded answer."
        )
    ),
    Document(
        page_content=(
            "Agentic RAG extends traditional RAG by allowing an AI agent to "
            "decide when retrieval is necessary and dynamically control the "
            "retrieval workflow."
        )
    ),
    Document(
        page_content=(
            "LangGraph is a framework for building stateful, multi-step agent "
            "workflows. Nodes represent processing steps and edges determine "
            "how information flows between them."
        )
    ),
]


def retrieve(state: AgentState):
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    results = vector_store.similarity_search(state["question"], k=2)

    context = "\n\n".join(doc.page_content for doc in results)

    return {"context": context}


def generate(state: AgentState):
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the provided context. "
                "If the context does not contain the answer, say you don't "
                "have enough information.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": state["context"],
            "question": state["question"],
        }
    )

    return {"answer": response.content}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    question = input("Ask a question: ")

    result = graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
        }
    )

    print("\nAnswer:")
    print(result["answer"])
