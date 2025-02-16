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

class app:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Constants
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", huggingfacehub_api_token=st.secrets["HUGGING_FACE_TOKEN", None])

        # Initialize Language Model
        self.llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"],
                            model_name="Gemma2-9b-It",
                            max_tokens=512  # Limit response size
                            )

        # Initialize Streamlit App
        st.title("Mel's RAG Bot")

        # Generate a unique session ID for each user
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = str(uuid.uuid4())  # Generates a unique ID per session

        # Initialize Session Storage for Chat History
        if "chat_store" not in st.session_state:
            st.session_state["chat_store"] = {}

        # Prompts
        self.CONTEXTUALIZE_PROMPT = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question to answer without the chat history."
            " Do not answer the question, just reformulate it if needed; otherwise, return it as is."
        )

        self.QA_SYSTEM_PROMPT = (
            "You are an assistant that helps with answering questions about documents. "
            "You are given a context and a question. "
            "You need to answer the question using only the context. "
            "If you don't know the answer, just say that you don't know. "
            "Do not try to make up an answer.\n\n"
            "Context: {context}"
        )

    def run(self):
        # Streamlit UI for file upload
        uploaded_files = st.file_uploader("Upload your PDF files", type=["txt", "csv", "pdf", "xls", "xlsx", "doc", "docx"], accept_multiple_files=True)

        if uploaded_files:
            # Process files and create retrieval chain
            document_chunks = self.process_uploaded_files(uploaded_files)
            retrieval_chain = self.get_retrieval_chain(document_chunks)

            # Create conversational chain with session memory
            conversational_rag_chain = self.create_conversational_chain(retrieval_chain)

            # User input section
            user_input = st.text_input("Ask a question:")

            if user_input:
                # Retrieve session history
                session_history = self.get_session_history(st.session_state["session_id"])

                # Invoke the conversational chain
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state["session_id"]}}
                )

                # Display chat history and response
                st.write("Assistant:", response["answer"])
                st.write("Chat History:", session_history.messages)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieve or initialize chat history for a session, keeping only recent messages."""
        if session_id not in st.session_state.chat_store:
            st.session_state.chat_store[session_id] = ChatMessageHistory()

        # Keep only last 5 messages to prevent exceeding context length
        history = st.session_state.chat_store[session_id]
        history.messages = history.messages[-10:]  # Keep only last 10 messages

        return history

    def process_uploaded_files(self, uploaded_files):
        """Load and split multiple PDF files into smaller chunks efficiently."""
        all_documents = []
        loader = None

        for uploaded_file in uploaded_files:
            # Create a unique temporary file for each upload with the original extension
            file_extension = os.path.splitext(uploaded_file.name)[1]  # Extract the file extension

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name  # Get the file path

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = UnstructuredFileLoader(temp_file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension == ".xlsx" or file_extension == ".xls":
                loader = UnstructuredExcelLoader(temp_file_path, mode="elements")
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = UnstructuredWordDocumentLoader(temp_file_path)

            docs = loader.load()
            all_documents.extend(docs)

            # Optionally: Delete temp file after loading (uncomment below)
            os.remove(temp_file_path)

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        return text_splitter.split_documents(all_documents)

    def get_retrieval_chain(self, docs):
        """Create a vectorstore-based retriever chain with history awareness."""
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        contextualize_prompt = ChatPromptTemplate.from_messages([("system", self.CONTEXTUALIZE_PROMPT), ("human", "{input}")])
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([("system", self.QA_SYSTEM_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        qa_doc_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_doc_chain)

        return retrieval_chain

    def create_conversational_chain(self, retrieval_chain):
        """Wrap the retrieval chain with session-based message history."""
        return RunnableWithMessageHistory(
            retrieval_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

# Instantiate and run the bot
if __name__ == "__main__":
    bot = app()
    bot.run()
