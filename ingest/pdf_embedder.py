import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models

from utils.model_cache import get_embedding_model

EMBED_DIM = 384

def embed_pdf_to_qdrant(pdf_path, collection_name="pdf_collection"):
    from qdrant_client.http import models

    try:
        # Load and chunk PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(f"No content extracted from PDF: {pdf_path}")

        # Embed using cached model
        model = get_embedding_model()
        embeddings = [model.encode(chunk.page_content) for chunk in chunks]

        # Qdrant
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        
        # Check if collection exists, if not create it
        try:
            client.get_collection(collection_name)
            # Collection exists, delete it to recreate with fresh data
            client.delete_collection(collection_name)
        except:
            # Collection doesn't exist, which is fine
            pass
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
        )

        # Prepare points with proper indexing
        points = [
            models.PointStruct(
                id=i,
                vector=embeddings[i],
                payload={
                    "text": chunks[i].page_content,
                    "source": os.path.basename(pdf_path)
                }
            )
            for i in range(len(chunks))
        ]

        # Upload points in batches if needed
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )

        print(f"✅ Embedded {len(points)} chunks into collection '{collection_name}'")
    except Exception as e:
        error_msg = f"Failed to embed PDF {pdf_path}: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e
