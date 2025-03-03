import tempfile
import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize Streamlit App
st.title("File Assistant")

# Generate a unique session ID for each user
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())  # Unique ID per session

# Track previous uploaded files
if "prev_uploaded_files" not in st.session_state:
    st.session_state["prev_uploaded_files"] = None

# Initialize Session Storage for Chat History
if "chat_store" not in st.session_state:
    st.session_state["chat_store"] = {}

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Language Model
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="Gemma2-9b-It",
    max_tokens=512  # Limit response size
)

# Prompts
CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question, "
    "formulate a standalone question to answer without the chat history."
    " Do not answer the question, just reformulate it if needed; otherwise, return it as is."
)

QA_SYSTEM_PROMPT = (
    "You are an assistant that helps with answering questions about documents. "
    "You are given a context and a question. "
    "You need to answer the question using only the context. "
    "If you don't know the answer, just say that you don't know. "
    "Do not try to make up an answer.\n\n"
    "Context: {context}"
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or initialize chat history for a session, keeping only recent messages."""
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()

    history = st.session_state.chat_store[session_id]
    history.messages = history.messages[-10:]  # Keep only last 10 messages
    return history


def process_uploaded_files(uploaded_files):
    """Load and split multiple files into smaller chunks efficiently."""
    all_documents = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = UnstructuredFileLoader(temp_file_path)
        elif file_extension == ".csv":
            loader = CSVLoader(temp_file_path)
        elif file_extension in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(temp_file_path, mode="elements")
        elif file_extension in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(temp_file_path)

        docs = loader.load()
        all_documents.extend(docs)
        os.remove(temp_file_path)

        # Calculate total text length
        total_text_length = sum(len(doc.page_content) for doc in docs)

        # Check if document meets the minimum chunk size
        if total_text_length < CHUNK_SIZE:
            st.error(
                f"The document '{uploaded_file.name}' is too small ({total_text_length} characters). It must be at least {CHUNK_SIZE} characters.")
            continue  # Skip this file and process others

        all_documents.extend(docs)

        # If no valid documents were found, return an empty list
        if not all_documents:
            return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(all_documents)


def get_retrieval_chain(docs):
    """Create a vectorstore-based retriever chain with history awareness."""
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    contextualize_prompt = ChatPromptTemplate.from_messages([("system", CONTEXTUALIZE_PROMPT), ("human", "{input}")])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", QA_SYSTEM_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    qa_doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_doc_chain)

    return retrieval_chain


def create_conversational_chain(retrieval_chain):
    """Wrap the retrieval chain with session-based message history."""
    return RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# Streamlit UI for file upload
uploaded_files = st.file_uploader("Upload your files",
                                  type=["txt", "csv", "pdf", "xls", "xlsx", "doc", "docx"],
                                  accept_multiple_files=True)

# Detect if files were removed
if st.session_state["prev_uploaded_files"] and not uploaded_files:
    st.session_state.clear()  # Clears session state when files are removed
    st.text_input("")
    st.rerun()  # Refreshes the UI to reset the chatbot

# Store current uploaded files in session state
st.session_state["prev_uploaded_files"] = uploaded_files

# Process files only once
if uploaded_files and "conversational_rag_chain" not in st.session_state:
    with st.spinner("Processing files..."):  # Show spinner
        document_chunks = process_uploaded_files(uploaded_files)
        retrieval_chain = get_retrieval_chain(document_chunks)
        conversational_rag_chain = create_conversational_chain(retrieval_chain)

    # Store in session state to avoid reprocessing
    st.session_state["conversational_rag_chain"] = conversational_rag_chain

# Text input for user query
user_input = st.text_input("Ask a question:")

if user_input:
    if "conversational_rag_chain" not in st.session_state:
        st.error("Please upload a file first.")
    else:
        session_history = get_session_history(st.session_state["session_id"])

        # Stream response
        response_stream = st.session_state["conversational_rag_chain"].stream(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state["session_id"]}}
        )

        # Display streamed response
        response_container = st.empty()
        streamed_text = ""

        for chunk in response_stream:
            chunk_text = chunk.get("answer") or chunk.get("text") or ""
            streamed_text += chunk_text
            response_container.write(streamed_text)