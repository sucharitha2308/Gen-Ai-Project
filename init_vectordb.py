import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    body = []
    reading_body = False
    for line in lines:
        if reading_body:
            body.append(line)
        elif line.strip() == "":
            reading_body = True
    return "\n".join(body).strip()

def auto_label(text):
    text = text.lower()
    financial_keywords = ['budget', 'invoice', 'purchase', 'financial', 'report', 'quarterly', 'bank', 'money', 'expense', 'cost', 'payment', 'transaction', 'audit', 'billing']
    urgent_keywords = ['urgent', 'immediate', 'emergency', 'deadline', 'breach', 'asap', 'critical', 'alert', 'warning', 'high priority', 'immediate action']
    hr_keywords = ['hr', 'policies', 'performance', 'review', 'insurance', 'promotion', 'holiday', 'leave', 'benefits', 'hiring', 'salary', 'recruitment', 'onboarding', 'resignation', 'interview']
    
    for word in urgent_keywords:
        if word in text:
            return 'Urgent'
    for word in financial_keywords:
        if word in text:
            return 'Financial'
    for word in hr_keywords:
        if word in text:
            return 'HR'
    return 'General'

def init_db():
    print("Initializing Vector Database...")
    
    # 1. Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    
    # Use a lightweight model for embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Create or get collection
    try:
        collection = chroma_client.create_collection(name="email_collection", embedding_function=sentence_transformer_ef)
    except Exception:
        # If it exists, delete and recreate to start fresh
        chroma_client.delete_collection(name="email_collection")
        collection = chroma_client.create_collection(name="email_collection", embedding_function=sentence_transformer_ef)

    # 2. Load Data
    csv_path = 'emails.csv'
    if not os.path.exists(csv_path):
        print("emails.csv not found!")
        return

    print("Loading emails...")
    # Load a subset for demonstration speed (e.g., 2000 emails)
    df = pd.read_csv(csv_path, nrows=2000)
    
    print("Processing data...")
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        content = parse_raw_message(row['message'])
        if not content.strip():
            continue
            
        category = auto_label(content)
        
        documents.append(content)
        metadatas.append({"category": category})
        ids.append(str(idx))
        
    print(f"Adding {len(documents)} documents to ChromaDB...")
    
    # Add in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Added batch {i}-{end}")

    print("Vector Database Initialized Successfully!")

if __name__ == "__main__":
    init_db()
