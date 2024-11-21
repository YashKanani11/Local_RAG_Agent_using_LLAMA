import streamlit as st
import os
import time
from rag_agent import load_documents, split_docs, embed_to_chromaDB, query_chromaDB

# Streamlit App
st.title("📚 RAG Agent with Llama 3.1 8B")
st.write("Upload your PDFs and ask questions to the agent!")

# Sidebar for file upload
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Process uploaded files
if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    st.sidebar.success("Files uploaded successfully!")

# Initialize session state for processing status
if "processing" not in st.session_state:
    st.session_state.processing = False

# Button to process documents
if st.button("Process Documents", disabled=st.session_state.processing):
    st.session_state.processing = True
    with st.spinner("Processing documents..."):
        start_time = time.time()
        try:
            documents = load_documents()
            chunks = split_docs(documents)
            new_chunks_count = embed_to_chromaDB(chunks)
            processing_time = time.time() - start_time
            st.success(f"Processed documents and added {new_chunks_count} new chunks to the database in {processing_time:.2f} seconds.")
        except Exception as e:
            st.error(f"An error occurred while processing the documents: {e}")
        finally:
            st.session_state.processing = False

# Query the agent
query_text = st.text_input("Ask a question:", disabled=st.session_state.processing)
if st.button("Get Answer", disabled=st.session_state.processing):
    if query_text:
        answer = query_chromaDB(query_text)
        st.write("### Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question!")
