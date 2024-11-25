__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.agents import Tool
from langchain_community.utilities import SerpAPIWrapper

CHROMA_PATH = 'chroma'

def load_documents(data_path="data"):
    """Load documents from a directory."""
    try:
        loader = PyPDFDirectoryLoader(data_path)
        return loader.load()
    except ImportError as e:
        print(f"ImportError: {e}")
        raise ImportError("Please ensure PyMuPDF and PyPDF2 are installed.")
    except Exception as e:
        print(f"Error while loading documents: {e}")
        raise

def split_docs(documents, cs=400, co=80):
    """Split documents into smaller chunks with specified chunk size and overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_id(chunks):
    """Assign unique chunk IDs to each chunk."""
    last_page_id = None
    curr_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        curr_page_id = f"{source}:{page}"
        
        if curr_page_id == last_page_id:
            curr_chunk_index += 1
        else:
            curr_chunk_index = 0
        
        chunk_id = f"{curr_page_id}:{curr_chunk_index}"
        last_page_id = curr_page_id
        chunk.metadata["chunk_id"] = chunk_id

    return chunks

def get_embeddings():
    """Return Ollama embeddings."""
    return OllamaEmbeddings(model="nomic-embed-text")

def embed_to_chromaDB(chunks: list[Document]):
    """Embed document chunks into ChromaDB."""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )
    chunks_with_ids = calculate_chunk_id(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["chunk_id"] not in existing_ids]
    if new_chunks:
        new_chunk_ids = [chunk.metadata["chunk_id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    return len(new_chunks)

def query_chromaDB(query_text: str):
    """Query ChromaDB for relevant context and generate a response."""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n --- \n\n".join([doc.page_content for doc, _ in results])

    prompt_temp = """ 
    Answer the question with the following context only:
    {context}

    ___
    Answer carefully with the above context: {ques}
    """
    prompt_temp_str = ChatPromptTemplate.from_template(prompt_temp)
    prompt = prompt_temp_str.format(context=context_text, ques=query_text)

    model = Ollama(model="llama3.1:8b")
    response_text = model.invoke(prompt)
    source_for_answer = [doc.metadata.get("chunk_id") for doc, _ in results]
    return f"{response_text}. Sources: {source_for_answer}"

def create_custom_tool():
    """Create a SerpAPI-based tool."""
    def custom_search(query: str):
        search = SerpAPIWrapper()
        site_specific_query = f"site:example.com {query}"
        return search.run(site_specific_query)

    return Tool(
        name="SpecificSiteSearch",
        func=custom_search,
        description="Searches a specific website for information."
    )
