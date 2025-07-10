import streamlit as st
import os
from src.document_processor import (
    load_and_chunk_file,
    create_and_store_embeddings,
    load_vector_store,
    is_supported_file  # <-- Add this utility function to your utils or document_processor
)
from src.agent_setup import setup_qa_chain
from src.utils import load_environment_variables

# Load environment variables
try:
    OPENAI_API_KEY = load_environment_variables()
except ValueError as e:
    st.error(f"Configuration Error: {e}")
    st.info("Please make sure you have a .env file with OPENAI_API_KEY set.")
    st.stop()

# Streamlit UI Setup
st.set_page_config(page_title="AI-Powered Document QA Agent", layout="wide")
st.title("📄 AI-Powered Document QA Agent")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a document (.pdf, .docx, .pptx, .csv,.xlsx)", type=["pdf", "docx", "pptx", "csv","xlsx"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if is_supported_file(uploaded_file.name):
            with st.spinner(f"Processing {file_extension.upper()} document..."):
                os.makedirs("data", exist_ok=True)
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    # Process document and create embeddings
                    chunks = load_and_chunk_file(file_path)
                    st.session_state.vector_store = create_and_store_embeddings(chunks, OPENAI_API_KEY)
                    st.session_state.agent = setup_qa_chain(st.session_state.vector_store, OPENAI_API_KEY, verbose=True)
                    st.session_state.doc_processed = True
                    st.success(f"{file_extension.upper()} document processed and agent is ready!")
                    st.info(f"Processed {len(chunks)} chunks from the document.")
                except Exception as e:
                    st.error(f"Error while processing the document: {e}")
                    st.session_state.doc_processed = False
        else:
            st.error("Unsupported file type. Please upload a .pdf, .docx, .pptx, or .csv file.")
    else:
        st.session_state.doc_processed = False
        st.session_state.vector_store = None
        st.session_state.agent = None

# Main content area
if st.session_state.doc_processed:
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question here:")

    if question:
        if st.session_state.agent:
            with st.spinner("Getting answer..."):
                try:
                    response = st.session_state.agent.invoke(question)
                    st.write("### Answer")
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    st.info("Please try re-uploading the document or check your OpenAI API key.")
        else:
            st.warning("Please upload and process a supported document first.")
else:
    st.info("Upload a document from the sidebar to get started.")

st.markdown("---")
# st.markdown("Built with ❤️ using Streamlit, LangChain, and OpenAI.")
