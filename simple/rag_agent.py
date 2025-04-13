from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.document.chunking.fixed import FixedSizeChunking
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.models.ollama import Ollama
from textwrap import dedent

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

embedder = SentenceTransformerEmbedder(
    id="cl-nagoya/sup-simcse-ja-base",
    dimensions=768
)

vector_db = PgVector(
    table_name="documents", 
    db_url=db_url, 
    search_type=SearchType.hybrid, 
    embedder=embedder,
)

knowledge_base = PDFKnowledgeBase(
    path="simple/data/",
    vector_db=vector_db,
    num_documents=10,
    chunking_strategy=FixedSizeChunking(
        chunk_size=200, 
        overlap=50, 
    )
)
# knowledge_base.load(recreate=True)

internal_document_agent = Agent(
    name="Internal Document Agent",
    model=Ollama(id="llama3.2:latest"),
    # model=OpenAIChat(id="gpt-4o-mini"),
    instructions=dedent("""\
    You are a helpful assistant. You are given a question and a context.
    You need to answer the question based on the context with the same language as the question.
    If you don't know the answer, just say I don't know with the same language as the question and don't make up an answer.
    Note: You have to answer directly to the question and don't give any other information.
    """),
    knowledge=knowledge_base,
    add_references=True,
    search_knowledge=False,
    debug_mode=True,
)
internal_document_agent.print_response("入札書の提出場所どころですか")
