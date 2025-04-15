from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import dotenv_values
import glob
import os
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

config = dotenv_values(".env")

def load_documents(directory="data"):
    """Load documents from a directory of PDF files."""
    pdf_files = glob.glob(os.path.join(directory, "*.PDF"))
    documents = []

    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_file)
            documents.extend(loader.load())
            print(f"Loaded document: {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    # Split documents into chunks
    text_splitter = SpacyTextSplitter(pipeline="ja_core_news_lg")

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} document chunks")
    return chunks

client = QdrantClient(url=config["QDRANT_URL"])
embeddings = HuggingFaceEmbeddings(model_name="cl-nagoya/sup-simcse-ja-base")

if not client.collection_exists(config["QDRANT_COLLECTION_NAME"]):
    client.create_collection(
        collection_name=config["QDRANT_COLLECTION_NAME"],
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

docs = load_documents("simple/data")

docsearch = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=config["QDRANT_URL"],
    api_key=config["QDRANT_API_KEY"],
    collection_name=config["QDRANT_COLLECTION_NAME"],
    prefer_grpc=True,
    force_recreate=not client.collection_exists(config["QDRANT_COLLECTION_NAME"]),
)


