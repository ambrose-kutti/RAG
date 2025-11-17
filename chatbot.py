import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from uuid import uuid4
import os

# Paths
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# Load and process documents
@st.cache_resource
def setup_rag():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    chunks = [doc for doc in chunks if doc.page_content.strip()]

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="rag_chatbot",
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )

    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = """
    You are a helpful AI assistant. Use the following context to answer the user's question.

    Context:
    {context}

    Question: {question}

    Instructions:
    - Answer based only on the provided context
    - If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided documents."
    - Keep answers concise and to the point. Focus on the key information. not full paragraphs

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = Ollama(model="llama3.2")

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, vector_store

rag_chain, vector_store = setup_rag()

# Streamlit UI
st.title("RAG Chatbot")
question = st.text_input("Ask a question based on your documents:")

if question:
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(question)
        st.success("Answer:")
        st.write(answer)

        docs = vector_store.similarity_search(question, k=1)
        st.markdown("### Retrieved Documents:")
        st.markdown(f"**Document 1:** {len(docs[0].page_content)} characters")
        for i, doc in enumerate(docs):
            st.markdown(f"**Document {i+1}:**")
            st.markdown(f"> {doc.page_content[:300].strip()}...")
            with st.expander(f"Document {i+1} Preview"):
                st.write(doc.page_content.strip())