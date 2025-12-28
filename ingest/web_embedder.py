"""
Web content embedding module for processing scraped web pages and storing in Qdrant.
"""
import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

from utils.model_cache import get_embedding_model

EMBED_DIM = 384

# Approximate token counts: 1 token ≈ 4 characters
# 500 tokens ≈ 2000 characters, 50 tokens ≈ 200 characters
CHUNK_SIZE = 2000  # characters (≈500 tokens)
CHUNK_OVERLAP = 200  # characters (≈50 tokens)
SIMILARITY_THRESHOLD = 0.95  # 95% similarity for deduplication


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def deduplicate_chunks(chunks: List[str], embeddings: List[List[float]], threshold: float = SIMILARITY_THRESHOLD) -> tuple:
    """
    Remove duplicate or highly similar chunks.
    
    Returns:
        (filtered_chunks, filtered_embeddings, kept_indices)
    """
    if len(chunks) <= 1:
        return chunks, embeddings, list(range(len(chunks)))
    
    filtered_chunks = []
    filtered_embeddings = []
    kept_indices = []
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        is_duplicate = False
        
        # Check similarity with already kept chunks
        for kept_emb in filtered_embeddings:
            similarity = cosine_similarity(emb, kept_emb)
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_chunks.append(chunk)
            filtered_embeddings.append(emb)
            kept_indices.append(i)
    
    return filtered_chunks, filtered_embeddings, kept_indices


def embed_web_to_qdrant(web_pages: List[Dict[str, str]], collection_name: str = "web_collection"):
    """
    Process web pages and embed them into Qdrant.
    
    Args:
        web_pages: List of dicts with keys: url, title, text, crawl_date, domain, source_type
        collection_name: Qdrant collection name
    """
    if not web_pages:
        raise ValueError("No web pages provided for embedding")
    
    try:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        all_chunks = []
        all_metadata = []
        
        for page in web_pages:
            text = page.get('text', '')
            if not text or len(text.strip()) < 50:
                continue
            
            # Split into chunks
            chunks = splitter.split_text(text)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'source_url': page.get('url', ''),
                    'page_title': page.get('title', ''),
                    'crawl_date': page.get('crawl_date', ''),
                    'domain': page.get('domain', ''),
                    'source_type': 'website'
                })
        
        if not all_chunks:
            raise ValueError("No text chunks extracted from web pages")
        
        # Generate embeddings
        model = get_embedding_model()
        embeddings = [model.encode(chunk) for chunk in all_chunks]
        
        # Deduplicate similar chunks
        filtered_chunks, filtered_embeddings, kept_indices = deduplicate_chunks(
            all_chunks, embeddings, threshold=SIMILARITY_THRESHOLD
        )
        
        # Update metadata to match filtered chunks
        filtered_metadata = [all_metadata[i] for i in kept_indices]
        
        print(f"📊 Original chunks: {len(all_chunks)}, After deduplication: {len(filtered_chunks)}")
        
        # Connect to Qdrant
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        
        # Check if collection exists, create if not
        try:
            client.get_collection(collection_name)
        except:
            # Collection doesn't exist, create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=EMBED_DIM,
                    distance=models.Distance.COSINE
                )
            )
        
        # Prepare points with metadata
        # Use a unique ID scheme: hash of url + chunk index
        import hashlib
        points = []
        for i, (chunk, emb, meta) in enumerate(zip(filtered_chunks, filtered_embeddings, filtered_metadata)):
            # Generate unique ID from URL + chunk index
            url_hash = hashlib.md5(meta['source_url'].encode()).hexdigest()[:8]
            point_id = int(f"{url_hash}{i}", 16) % (2**63)  # Convert to int64
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={
                        "text": chunk,
                        "source": meta['source_url'],  # Keep 'source' for compatibility
                        "source_url": meta['source_url'],
                        "page_title": meta['page_title'],
                        "crawl_date": meta['crawl_date'],
                        "domain": meta['domain'],
                        "source_type": meta['source_type']
                    }
                )
            )
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
        
        print(f"✅ Embedded {len(points)} web chunks into collection '{collection_name}'")
        return len(points)
        
    except Exception as e:
        error_msg = f"Failed to embed web content: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

