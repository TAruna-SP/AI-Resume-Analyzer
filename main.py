from pypdf import PdfReader
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openAI_Embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Document Ingestion/loading
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text

# Chunking
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks

def get_embeddings(text_list):
    response = openAI_Embedding_client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings).astype("float32")    

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index

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