import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# List of supported file extensions
SUPPORTED_FORMATS = [".pdf", ".csv", ".docx", ".pptx", ".xlsx"]

# Function to check if file format is supported
def is_supported_file(file_path):
    return os.path.splitext(file_path)[1].lower() in SUPPORTED_FORMATS

# Loader based on file extension
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif ext == ".csv":
        loader = CSVLoader(file_path)
        return loader.load()
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()
    elif ext == ".xlsx":
        return load_excel_as_documents(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Custom function to load Excel as documents
def load_excel_as_documents(file_path):
    excel_data = pd.read_excel(file_path, sheet_name=None)  # Load all sheets
    documents = []

    for sheet_name, df in excel_data.items():
        content = f"Sheet: {sheet_name}\n\n{df.to_string(index=False)}"
        documents.append(Document(page_content=content, metadata={"source": file_path, "sheet": sheet_name}))

    return documents

# Load and chunk a file into smaller text pieces
def load_and_chunk_file(file_path):
    if not is_supported_file(file_path):
        raise ValueError("Unsupported file format")

    documents = load_document(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create FAISS index from chunks and store locally
def create_and_store_embeddings(chunks, openai_api_key, db_path="faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(db_path)
    return vector_store

# Load the FAISS vector store
def load_vector_store(openai_api_key, db_path="faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store
