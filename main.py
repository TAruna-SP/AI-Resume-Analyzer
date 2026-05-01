from pypdf import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Use Sentence Transformers for local Embeddings as Open AI/Gemini APIs not working.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Document Ingestion/loading
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text

# Chunking
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks

# Convert chunks to Embeddings
def get_embeddings(text_list):
    embeddings = embedding_model.encode(text_list)
    return np.array(embeddings).astype("float32")

# Store and similarity search using faiss library for embeddings
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index

# convert query to embeddings 
def get_query_embedding(query):
    embedding = embedding_model.encode([query])
    return np.array(embedding[0]).astype("float32")

# Search in faiss local library for similar embeddings/vectors as the original query
def retrieve_chunks(query, index, chunks, top_k=3):
    query_vector = get_query_embedding(query)
    
    query_vector = np.expand_dims(query_vector, axis=0)
    
    distances, indices = index.search(query_vector, top_k)
    
    print("Scores:", distances)
    results = [chunks[i] for i in indices[0]]
    
    return results        

# LLM using Groq
def generate_answer(query, retrieved_chunks):
    
    context = "\n\n".join(retrieved_chunks[:3])

    prompt = f"""
You are an expert AI resume analyzer.
Instructions:

- Only use the provided context
- Do NOT hallucinate
- If answer not found, say "Not mentioned"


Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",   
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Main
if __name__ == "__main__":
    file_path = "Aruna-21f3000883-IITM BS.pdf"   # put your resume here
    text = extract_text_from_pdf(file_path)

    chunks = chunk_text(text)
    
    print("Total chunks:", len(chunks))
    print("\nFirst chunk:\n")
    print(chunks[0])
    # print(text[:1000])  # print first 1000 characters

    embeddings = get_embeddings(chunks)
    
    print("Embeddings shape:", embeddings.shape)
    
    index = create_faiss_index(embeddings)
    
    print("FAISS index created successfully!")

    # query = "What are the candidate's skills?"

    # results = retrieve_chunks(query, index, chunks)

    # print("\nTop relevant chunks:\n")
    # for i, chunk in enumerate(results):
    #     print(f"\n--- Chunk {i+1} ---\n")
    #     print(chunk)

    query = input("Ask a question: ")

    retrieved_chunks = retrieve_chunks(query, index, chunks)

    answer = generate_answer(query, retrieved_chunks)

    print("\nFinal Answer:\n")
    print(answer)    