from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import aiohttp
import tempfile
import os
import sqlite3
import json
from groq import Groq
from crawler import DocumentProcessor, NLPProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Query-Retrieval System",
    description="An intelligent system for processing documents and answering questions using data from PDFs and crawled sources.",
    version="1.0.0",
    root_path="/api/v1"
)

# Authentication setup
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    correct_token = "225bc5a220c1e087a557d9738720c84d5a1dca03fd966f83977f563fe741968d"
    if credentials.credentials != correct_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Request model
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Successfully initialized Groq client")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    raise Exception(f"Groq client initialization failed: {str(e)}")

# Utility functions
async def download_file(url: str) -> str:
    """Download a file from a URL to a temporary location."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download document")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(await response.read())
                return tmp_file.name

def split_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately chunk_size characters."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_relevant_chunks(question: str, vectorizer: TfidfVectorizer, chunk_vectors, chunks: List[str], top_k: int = 3) -> str:
    """Retrieve the top-k most relevant chunks for a question using pre-fitted TF-IDF."""
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return " ".join([chunks[i] for i in top_indices])

def get_excerpt(content: str, entity_text: str) -> str:
    """Extract the first sentence from content that contains the entity_text."""
    sentences = sent_tokenize(content)
    for sentence in sentences:
        if entity_text.lower() in sentence.lower():
            return sentence
    return ""

def get_kg_info(kg: dict, entity_text: str) -> str:
    """Retrieve relationships for the given entity from the knowledge graph."""
    entity_id = None
    for eid, data in kg['entities'].items():
        if data['text'].lower() == entity_text.lower():
            entity_id = eid
            break
    if entity_id:
        relationships = [rel for rel in kg['relationships'] if rel['source'] == entity_id or rel['target'] == entity_id]
        if relationships:
            info = f"Entity '{entity_text}' has relationships: "
            for rel in relationships:
                source_text = kg['entities'][rel['source']]['text']
                target_text = kg['entities'][rel['target']]['text']
                related_entity = target_text if rel['source'] == entity_id else source_text
                info += f"{rel['relation']} with {related_entity}, "
            return info.rstrip(", ")
    return ""

def generate_answer(context: str, question: str, max_tokens: int = 512) -> str:
    """Generate an answer using the Groq API with llama-3.3-70b-versatile model."""
    try:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with expertise in insurance, legal, HR, and compliance domains. Provide accurate and concise answers based on the given context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        # Use 'content' instead of 'syscontent'
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer for question '{question}': {str(e)}")
        return "Unable to generate answer due to an error."

# Endpoint definitions
@app.post("/hackrx/run")
async def run_query(request: QueryRequest, token: str = Depends(verify_token)):
    """Process a document and answer questions based on its content and crawled data."""
    # Download and process the document
    pdf_path = None
    try:
        pdf_path = await download_file(request.documents)
        doc_processor = DocumentProcessor()
        text = doc_processor.process_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)

    # Split text into manageable chunks
    chunks = split_text(text, chunk_size=1000)

    # Initialize and fit TF-IDF vectorizer once
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(chunks)

    # Initialize NLPProcessor
    nlp_processor = NLPProcessor()

    # Connect to the database
    try:
        db_path = os.path.join('bajajfinserv_data', 'bajajfinserv.db')
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to connect to database")

    # Load knowledge graph
    try:
        kg_path = os.path.join('bajajfinserv_data', 'knowledge_graph.json')
        with open(kg_path, 'r') as f:
            kg = json.load(f)
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load knowledge graph")

    # Process each question
    answers = []
    for question in request.questions:
        # Extract entities from the question
        entities = nlp_processor.extract_entities(question)
        entity_texts = [entity.text for entity in entities]

        # Get web content excerpts from bajajfinserv.db
        excerpts = []
        for entity_text in entity_texts:
            cursor = conn.cursor()
            cursor.execute("SELECT title, content FROM web_content WHERE content LIKE ?", ('%' + entity_text + '%',))
            results = cursor.fetchall()[:3]  # Limit to 3 results per entity
            for title, content in results:
                excerpt = get_excerpt(content, entity_text)
                if excerpt:
                    excerpts.append(f"From '{title}': {excerpt}")

        # Get knowledge graph info from knowledge_graph.json
        kg_info = []
        for entity_text in entity_texts:
            info = get_kg_info(kg, entity_text)
            if info:
                kg_info.append(info)

        # Get TF-IDF relevant chunks from the PDF
        tfidf_context = get_relevant_chunks(question, vectorizer, chunk_vectors, chunks)

        # Combine all context
        additional_context = "\n".join(excerpts) + "\n" + "\n".join(kg_info)
        full_context = f"PDF content: {tfidf_context}\n\nAdditional information: {additional_context}"

        # Generate answer
        try:
            answer = generate_answer(full_context, question)
            answers.append(answer)
        except Exception as e:
            logger.error(f"Error generating answer for question '{question}': {str(e)}")
            answers.append("Unable to generate answer due to an error.")

    # Close database connection
    conn.close()

    return {"answers": answers}

@app.get("/hackrx/run")
async def get_run_info():
    """GET endpoint for /hackrx/run (for browser or health check)."""
    return {"status": "API is running"}

@app.get("/favicon.ico")
async def favicon():
    """Serve a favicon.ico file if present, else return 204 No Content."""
    favicon_path = "assets/favicon.ico"
    if os.path.exists(favicon_path):
        from fastapi.responses import FileResponse
        return FileResponse(favicon_path, media_type="image/x-icon")
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/")
async def root():
    """Root endpoint for health check or welcome message."""
    return {"message": "Welcome to the LLM-Powered Query-Retrieval System. Use /api/v1/hackrx/run for queries."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)