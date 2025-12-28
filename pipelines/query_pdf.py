import os
from qdrant_client import QdrantClient

from utils.groq_client import get_groq_client
from utils.model_cache import get_embedding_model

EMBED_DIM = 384

def ask_pdf(question: str, collections: list, top_k=6, return_chunks=False):
    if not collections:
        raise ValueError("No collections provided for querying")
    
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        model = get_embedding_model()
        q_emb = model.encode(question)

        all_hits = []
        for collection in collections:
            try:
                # Use query_points instead of search (newer Qdrant API)
                response = qdrant.query_points(
                    collection_name=collection,
                    query=q_emb,  # Pass vector directly as query
                    limit=top_k
                )
                all_hits.extend(response.points)
            except Exception as e:
                raise RuntimeError(f"Error querying collection '{collection}': {str(e)}") from e

        if not all_hits:
            return "I cannot find this information in the uploaded documents or websites."

        # Sort by score and pick top overall
        sorted_hits = sorted(all_hits, key=lambda h: h.score, reverse=True)[:top_k]

        # Format context with source information
        context_parts = []
        sources_used = set()
        
        for h in sorted_hits:
            payload = h.payload
            source_type = payload.get('source_type', 'pdf')
            
            # Determine source title/name
            if source_type == 'website':
                source_title = payload.get('page_title', payload.get('source_url', 'Website'))
                source_url = payload.get('source_url', '')
                source_info = f"[Source: {source_title}]"
                sources_used.add((source_title, source_url, 'website'))
            else:
                source_name = payload.get('source', 'PDF')
                source_info = f"[Source: {source_name}]"
                sources_used.add((source_name, None, 'pdf'))
            
            text = payload.get('text', '')
            context_parts.append(f"{source_info} {text}")
        
        context = "\n---\n".join(context_parts)
        
        # Format sources list for citation
        sources_list = []
        for name, url, stype in sources_used:
            if url:
                sources_list.append(f"- {name} ({url})")
            else:
                sources_list.append(f"- {name}")

        prompt = f"""You are a RAG-powered assistant answering questions from PDFs and websites.

STRICT RULES:
1. Answer ONLY using the provided CONTEXT below
2. If the answer isn't in CONTEXT, respond: "I cannot find this information in the uploaded documents or websites."
3. Never use external knowledge or make assumptions
4. Always cite sources with format: [Source: {{title}}]

RESPONSE FORMAT:
- Start with direct answer (2-3 sentences)
- Add brief explanation if needed
- End with "Sources:" and list each unique source used

CONTEXT:
{context}

USER QUESTION: {question}

Answer:"""
        
        groq = get_groq_client()
        # Use environment variable for model, with fallback to current available models
        # Try models in order: user preference -> stable models
        user_model = os.getenv("GROQ_MODEL")
        fallback_models = [
            "llama-3.1-8b-instant",      # Fast, stable 8B model (most reliable)
            "llama-3.3-70b-versatile",   # Latest 70B model (if available)
            "mixtral-8x7b-32768",        # Alternative model
            "llama-3.1-70b-versatile"    # Previous version (fallback)
        ]
        
        model_name = user_model if user_model else fallback_models[0]
        
        # Try the model, with fallback if it's decommissioned
        try:
            resp = groq.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
        except Exception as e:
            error_str = str(e)
            # If model is decommissioned or invalid, try fallback models
            if ("decommissioned" in error_str.lower() or "not found" in error_str.lower() or "invalid" in error_str.lower()) and model_name not in fallback_models[1:]:
                for fallback_model in fallback_models[1:]:
                    try:
                        resp = groq.chat.completions.create(
                            model=fallback_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=1000
                        )
                        # Success! Use this model
                        model_name = fallback_model
                        break
                    except Exception as fallback_error:
                        # Try next fallback
                        continue
                else:
                    # All models failed, raise a helpful error
                    raise RuntimeError(
                        f"❌ All Groq models failed. The model '{model_name}' is not available.\n\n"
                        f"**Original error:** {error_str}\n\n"
                        f"**Solution:** Please set a valid GROQ_MODEL in your .env file.\n"
                        f"Check available models at: https://console.groq.com/docs/models\n"
                        f"Common working models: llama-3.1-8b-instant, llama-3.3-70b-versatile, mixtral-8x7b-32768"
                    ) from e
            else:
                # Re-raise if it's a different error or we've exhausted fallbacks
                raise

        answer = resp.choices[0].message.content
        
        # Ensure sources are mentioned if not already in answer
        if sources_list and "Sources:" not in answer:
            answer += f"\n\n**Sources:**\n" + "\n".join(sources_list)
        
        if return_chunks:
            # Format chunks for debug display
            chunks_data = []
            for idx, h in enumerate(sorted_hits):
                payload = h.payload
                chunk_data = {
                    'text': payload.get('text', ''),
                    'score': h.score,
                    'chunk_index': idx + 1,
                    'source_name': payload.get('page_title') or payload.get('source', 'Unknown'),
                    'source_url': payload.get('source_url', ''),
                    'timestamp': payload.get('crawl_date', 'N/A')
                }
                chunks_data.append(chunk_data)
            
            return answer, chunks_data
        
        return answer
    except Exception as e:
        raise RuntimeError(f"Error processing query: {str(e)}") from e
