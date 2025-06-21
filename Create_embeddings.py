import json
import os
import re
from llama_parse import LlamaParse
from config import *
from tqdm import tqdm
import time

# Parser setup - moved to top
parser = LlamaParse(
    api_key=LLAMAPARSE_API_KEY, 
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en",
)

def chunk_for_rag(text: str, target_tokens: int = 350, overlap: int = 75):
    # Clean text
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'(?:Page \d+.*|^\s*\d+\s*$)', '', text, flags=re.MULTILINE)
    
    # Find section boundaries
    patterns = r'^\s*(?:\d+\.[\d\.]*\s+[A-Z]|(?:CREDIT|PREREQUISITE|FEATURE|REQUIREMENT|INTENT|REQUIREMENTS|COMPLIANCE|VERIFICATION|DOCUMENTATION|Exception|TABLE|Figure|Formula)\s*[A-Z0-9]*[:\s]*[A-Z]|[A-Z]{1,3}\d*[a-z]?\d*\s*[:-]\s*[A-Z])[^\n]*$'
    
    boundaries = [0] + [m.start() for m in re.finditer(patterns, text, re.MULTILINE)] + [len(text)]
    sections = [text[boundaries[i]:boundaries[i+1]].strip() for i in range(len(boundaries)-1)]
    
    chunks = []
    for section in sections:
        if not section or len(section.split()) < 10: continue
        
        if len(section.split()) <= target_tokens:
            chunks.append(section)
        else:
            # Split large sections
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            current, tokens = "", 0
            
            for para in paragraphs:
                para_tokens = len(para.split())
                if tokens + para_tokens <= target_tokens:
                    current += f"\n\n{para}" if current else para
                    tokens += para_tokens
                else:
                    if current: chunks.append(current)
                    # Start new chunk with overlap
                    sentences = re.split(r'(?<=[.!?])\s+', current)
                    overlap_text = " ".join(sentences[-overlap//20:]) if sentences else ""
                    current = f"{overlap_text}\n\n{para}" if overlap_text else para
                    tokens = len(current.split())
            
            if current: chunks.append(current)
    
    return [c for c in chunks if len(c.split()) >= 30 and sum(1 for ch in c if ch.isalpha()) > len(c) * 0.4]

def get_embedding(text, model=embedding_model, max_retries=3):
    for attempt in range(max_retries):
        try:
            return local_client.embeddings.create(input=[text.replace("\n", " ")], model=model).data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2 ** attempt)
            else: raise e

def process_pdfs_and_create_embeddings(directory="knowledge_pool"):
    for filename in os.listdir(directory):
        if not filename.endswith(".pdf"): continue
        
        try:
            print(f"Processing {filename}...")
            
            # Parse PDF and extract text
            with open(os.path.join(directory, filename), "rb") as f:
                docs = parser.load_data(f, extra_info={"file_name": os.path.join(directory, filename)})
            
            full_text = "\n".join(doc.text for doc in docs if doc.text.strip())
            if not full_text: continue
            
            # Save text file
            base_name = os.path.splitext(filename)[0]
            with open(os.path.join(directory, f"{base_name}.txt"), 'w', encoding='utf-8', errors='replace') as f:
                f.write(full_text)
            
            # Create chunks and embeddings
            chunks = chunk_for_rag(full_text)
            print(f"Created {len(chunks)} chunks from {filename}")
            
            embeddings = []
            for chunk in tqdm(chunks, desc=f"Embedding {filename}"):
                clean_chunk = re.sub(r'\n+', ' ', chunk).strip()
                embeddings.append({'content': clean_chunk, 'vector': get_embedding(clean_chunk)})
            
            # Save embeddings
            with open(os.path.join(directory, f"{base_name}.json"), 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, indent=2, ensure_ascii=False)
            
            print(f"Finished {filename} - {len(embeddings)} embeddings created")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_pdfs_and_create_embeddings()