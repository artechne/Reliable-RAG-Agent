import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function

PDF_DATA_PATH = "data"
CHROMA_DB_PATH = "chroma"

def load_documents():
    document_loader = PyPDFDirectoryLoader(PDF_DATA_PATH, "*.PDF")
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        #TODO: make these parameters configurable and assign best values
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma_db(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_DB_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    exisiting_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(exisiting_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in exisiting_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_idx = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_idx += 1
        else:
            current_chunk_idx = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_idx}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks

def get_chroma_db():
    db = Chroma(persist_directory=CHROMA_DB_PATH, 
                embedding_function=get_embedding_function()
                )
    return db

def clear_chroma_db():
    if os.path.exists(CHROMA_DB_PATH):
        print("There is a existed chroma db, deleting it.")
        shutil.rmtree(CHROMA_DB_PATH)