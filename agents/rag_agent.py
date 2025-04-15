from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.document.chunking.fixed import FixedSizeChunking
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.models.ollama import Ollama
from textwrap import dedent
from dotenv import load_dotenv
import os

load_dotenv(override=True)

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
    path="agents/data",
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
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    instructions=dedent(
        """\
    あなたは役立つアシスタントです。質問とコンテキストが与えられます。
    質問と同じ言語でコンテキストに基づいて質問に答える必要があります。
    答えがわからない場合は、質問と同じ言語で「わかりません」と言うだけで、答えを作り上げないでください。
    注意：質問に直接答え、他の情報は提供しないでください。
    """
    ),
    knowledge=knowledge_base,
    add_references=True,
    search_knowledge=False,
    debug_mode=True,
)
# internal_document_agent.print_response("入札書の提出場所どころですか")
