"""
Enhanced RAG Query Pipeline with Query Enhancement and Improved Prompts.

This module provides:
1. Query enhancement using Groq LLM (optional, falls back to original query if unavailable)
2. Semantic search across multiple collections
3. Comprehensive answer generation with proper source attribution
"""

import os
from typing import List, Dict, Optional, Tuple, Any
from qdrant_client import QdrantClient

from utils.groq_client import get_groq_client
from utils.model_cache import get_embedding_model
from utils.query_enhancer import enhance_query, get_document_context_string

EMBED_DIM = 384

# TOKEN BUDGET MANAGEMENT
MAX_CHUNKS = 5  # Reduced from 10 to stay well within limits
MAX_CHUNK_LENGTH = 800  # Reduced from 1000 for tighter control
MAX_CONTEXT_TOKENS = 2000  # Rough estimate for total context


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token"""
    return len(text) // 4


def retrieve_chunks(
    query: str,
    collections: List[str],
    top_k: int = 10
) -> Tuple[List[Dict], List[Any]]:
    """
    Retrieve relevant chunks from Qdrant collections.
    
    Args:
        query: The search query (original or enhanced)
        collections: List of collection names to search
        top_k: Number of top results to retrieve
    
    Returns:
        Tuple of (formatted_chunks, raw_hits)
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    
    model = get_embedding_model()
    q_emb = model.encode(query)
    
    all_hits = []
    for collection in collections:
        try:
            response = qdrant.query_points(
                collection_name=collection,
                query=q_emb,
                limit=top_k
            )
            all_hits.extend(response.points)
        except Exception as e:
            raise RuntimeError(f"Error querying collection '{collection}': {str(e)}") from e
    
    # Sort by score and pick top overall
    sorted_hits = sorted(all_hits, key=lambda h: h.score, reverse=True)[:top_k]
    
    # Format chunks with metadata
    chunks = []
    for idx, hit in enumerate(sorted_hits):
        payload = hit.payload
        source_type = payload.get('source_type', 'pdf')
        
        if source_type == 'website':
            source_name = payload.get('page_title', payload.get('source_url', 'Website'))
            source_url = payload.get('source_url', '')
        else:
            source_name = payload.get('source', 'PDF Document')
            source_url = None
        
        chunks.append({
            'text': payload.get('text', ''),
            'source': source_name,
            'source_url': source_url,
            'source_type': source_type,
            'score': hit.score,
            'chunk_index': idx + 1,
            'timestamp': payload.get('crawl_date', 'N/A')
        })
    
    return chunks, sorted_hits


def truncate_chunks_to_budget(
    chunks: List[Dict], 
    max_total_tokens: int = MAX_CONTEXT_TOKENS
) -> List[Dict]:
    """
    Intelligently truncate chunks to fit within token budget.
    Prioritizes higher-scoring chunks and truncates text as needed.
    """
    truncated = []
    total_tokens = 0
    
    for chunk in chunks:
        text = chunk['text']
        chunk_tokens = estimate_tokens(text)
        
        if total_tokens + chunk_tokens > max_total_tokens:
            # Calculate remaining token budget
            remaining_tokens = max_total_tokens - total_tokens
            
            if remaining_tokens > 50:  # Only add if meaningful content fits
                chars_to_keep = remaining_tokens * 4
                truncated_text = text[:chars_to_keep] + "... [truncated]"
                truncated.append({**chunk, 'text': truncated_text})
            break
        
        # Truncate individual chunks if they're too long
        if chunk_tokens > MAX_CHUNK_LENGTH // 4:
            text = text[:MAX_CHUNK_LENGTH] + "... [truncated]"
        
        truncated.append({**chunk, 'text': text})
        total_tokens += estimate_tokens(text)
    
    return truncated


def build_context_string(chunks: List[Dict]) -> str:
    """Build formatted context string from chunks with source metadata."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk['source']
        source_type = chunk.get('source_type', 'document')
        text = chunk['text']
        context_parts.append(
            f"[CHUNK {i+1}] (Source: {source} | Type: {source_type})\n{text}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def enforce_bullet_length(text: str, max_words: int = 15) -> str:
    """Truncate bullet points that are too long to improve readability."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text


def improve_readability(answer: str) -> str:
    """Post-process answer to clean up any inline source citations."""
    # Remove any inline [Source: ...] citations that might have been added
    # Sources should only be listed at the end in the Sources section
    import re
    # Remove inline source citations from the body text
    cleaned = re.sub(r'\s*\[Source:[^\]]+\]', '', answer)
    return cleaned


def build_sources_list(chunks: List[Dict]) -> List[str]:
    """Build deduplicated sources list."""
    sources_seen = set()
    sources_list = []
    
    for chunk in chunks:
        source_key = (chunk['source'], chunk.get('source_url', ''))
        if source_key not in sources_seen:
            sources_seen.add(source_key)
            if chunk.get('source_url'):
                sources_list.append(f"- {chunk['source']} ({chunk['source_url']})")
            else:
                sources_list.append(f"- {chunk['source']}")
    
    return sources_list


def get_enhanced_system_prompt(
    context_text: str,
    source_list: str,
    source_type_list: str,
    num_sources: int,
    original_query: str
) -> str:
    """
    Return ultra-readable system prompt optimized for scanning and comprehension.
    Token count: ~1,000 tokens
    """
    return f"""You are a DOCUMENT-GROUNDED INTELLIGENCE SYSTEM. Answer questions using ONLY the retrieved context below.

═══════════════════════════════════════════════════════════════
RETRIEVED CONTEXT:
{context_text}

DOCUMENT SOURCES: {source_list}
USER QUESTION: {original_query}
═══════════════════════════════════════════════════════════════

CORE RULES (NON-NEGOTIABLE):

1. PRIMARY SOURCE: DOCUMENTS FIRST
   • ALWAYS check retrieved context FIRST
   • If information is found in documents → use it
   • Document information takes priority

2. SECONDARY SOURCE: GENERAL KNOWLEDGE (when documents don't have answer)
   • If the answer is NOT in retrieved context → you MAY use well-established general knowledge
   • BUT: Only use information that is widely accepted and verifiable
   • Clearly indicate when information comes from general knowledge vs documents
   • Be cautious and fact-check mentally before stating facts

3. VERIFICATION REQUIREMENT
   • Before providing information not in documents, verify it's accurate
   • Only state facts you are confident about
   • If uncertain about external information, say so explicitly
   • Prefer stating limitations over incorrect information

4. PERSON & IDENTITY QUERIES
   When asked about a PERSON (who is X, tell me about X, X's profile):
   
   ✅ From documents: Answer if context contains explicit biographical text
   ✅ From general knowledge: If not in documents, you MAY provide well-known biographical information (e.g., historical figures, famous people)
   ❌ Do NOT infer from: usernames, URLs, platform features alone
   
   Always clearly distinguish: "Based on the documents:" vs "From general knowledge:"

3. READABILITY REQUIREMENTS
   • Use simple, clear language
   • Write naturally - NO inline source citations
   • All sources listed ONLY at the end in Sources section
   • NO [Source: ...] tags in the text body

4. NO FILLER LANGUAGE
   NEVER say:
   • "Unfortunately, the provided context..."
   • "It appears that..."
   • "You may consider..."
   • "I recommend..."
   • "The context does not seem to..."

5. NO SPECULATION OR INFERENCE
   • Do NOT infer occupation from repository names
   • Do NOT assume background from project types
   • Do NOT guess location from context clues
   • Do NOT synthesize a bio from scattered data

OUTPUT FORMAT (MANDATORY - FOLLOW EXACTLY):

IF ANSWER IS IN DOCUMENTS:

### Summary
[Write 2-3 clear paragraphs summarizing the key points from the documents. Write naturally without inline citations.]

### Key Details

**[Category 1 - Short Name]**
[Write paragraphs explaining the details from the documents. Write naturally.]

**[Category 2 - Short Name]**
[More details from documents.]

### Sources
- **[Document Name]** - Used for: [summary, key details about X, information about Y]
- **[Another Document]** - Used for: [information about Z, details about W]

---

IF ANSWER IS NOT IN DOCUMENTS BUT YOU CAN PROVIDE GENERAL KNOWLEDGE:

### Summary
[Write summary based on general knowledge. Start with: "While this information is not in the provided documents, from general knowledge:"]

### Key Details
[Provide well-established, verifiable information. Clearly indicate this comes from general knowledge, not the documents.]

### Sources
- **General Knowledge** - Information not present in uploaded documents
- Note: The provided documents were checked first but did not contain this information

---

IF YOU ARE UNCERTAIN ABOUT EXTERNAL INFORMATION:

I cannot find information about [topic] in the provided documents. While I have some general knowledge about this topic, I want to ensure accuracy. Could you please verify the information from a reliable source?

IMPORTANT: Do NOT put [Source: ...] after every sentence. Write naturally and list all sources at the end. Always clearly distinguish between document-based and general knowledge information.

---

YOUR MISSION:
Make answers SCANNABLE. Users should grasp the main points in 30 seconds.
Think: "How would I want to read this on mobile?"

Now answer following ALL rules above:"""


def generate_answer(
    original_query: str,
    enhanced_query: str,
    chunks: List[Dict],
    sources_list: List[str]
) -> str:
    """
    Generate a comprehensive answer using Groq LLM with strict token management.
    
    Args:
        original_query: User's original question
        enhanced_query: Enhanced/optimized query
        chunks: Retrieved context chunks
        sources_list: Formatted list of sources
    
    Returns:
        Generated answer string
    """
    # Step 1: Truncate chunks to fit token budget
    safe_chunks = truncate_chunks_to_budget(chunks, max_total_tokens=MAX_CONTEXT_TOKENS)
    
    print(f"📦 Token-safe: Using {len(safe_chunks)}/{len(chunks)} chunks")
    
    # Step 2: Build formatted context
    context_text = build_context_string(safe_chunks)
    
    # Step 3: Extract source metadata
    sources_seen = set()
    unique_sources = []
    for chunk in safe_chunks:
        source_key = (chunk['source'], chunk.get('source_type', 'document'))
        if source_key not in sources_seen:
            sources_seen.add(source_key)
            source_type = chunk.get('source_type', 'document')
            unique_sources.append(f"{chunk['source']} ({source_type})")
    source_list = ", ".join(unique_sources)
    
    # Count source types
    source_types = {}
    for chunk in safe_chunks:
        stype = chunk.get('source_type', 'document')
        source_types[stype] = source_types.get(stype, 0) + 1
    source_type_list = ", ".join([f"{count} {stype}(s)" for stype, count in source_types.items()])
    
    # Step 4: Build enhanced prompt (1000 tokens)
    prompt = get_enhanced_system_prompt(
        context_text=context_text,
        source_list=source_list,
        source_type_list=source_type_list,
        num_sources=len(unique_sources),
        original_query=original_query
    )
    
    # Step 5: Verify token budget
    prompt_tokens = estimate_tokens(prompt)
    print(f"📊 Estimated prompt tokens: {prompt_tokens}")
    
    if prompt_tokens > 4000:  # Groq free tier allows ~6000 input tokens
        print("⚠️ Prompt too large, using emergency single-chunk mode")
        # Emergency fallback: use only the top chunk
        safe_chunks = safe_chunks[:1]
        context_text = build_context_string(safe_chunks)
        prompt = get_enhanced_system_prompt(
            context_text=context_text,
            source_list=safe_chunks[0]['source'],
            source_type_list=safe_chunks[0].get('source_type', 'document'),
            num_sources=1,
            original_query=original_query
        )
    
    # Step 6: Get Groq client and select model
    groq = get_groq_client()
    
    user_model = os.getenv("GROQ_MODEL")
    fallback_models = [
        "llama-3.1-8b-instant",      # Fast, efficient
        "llama-3.3-70b-versatile",   # More capable
        "mixtral-8x7b-32768",        # Large context
    ]
    
    model_name = user_model if user_model else fallback_models[0]
    models_to_try = [model_name] + [m for m in fallback_models if m != model_name]
    
    # Step 7: Try generation with fallbacks
    resp = None
    last_error = None
    
    for model in models_to_try:
        try:
            resp = groq.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            print(f"✅ Successfully used model: {model}")
            break
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Authentication errors
            if "401" in str(e) or "invalid_api_key" in error_str or "unauthorized" in error_str:
                raise RuntimeError(
                    f"❌ Groq API authentication failed.\n\n"
                    f"**Error:** Invalid or missing API key (401)\n\n"
                    f"**Solution:** Set valid GROQ_API_KEY in .env file.\n"
                    f"Get key from: https://console.groq.com/keys"
                ) from e
            
            # Token limit errors
            if "413" in str(e) or "request too large" in error_str or "tokens per minute" in error_str:
                raise RuntimeError(
                    f"❌ Request exceeds Groq token limits.\n\n"
                    f"**Error:** Too many tokens (free tier: 6000 TPM)\n\n"
                    f"**Solutions:**\n"
                    f"1. Query fewer documents\n"
                    f"2. Use shorter queries\n"
                    f"3. Upgrade tier: https://console.groq.com/settings/billing"
                ) from e
            
            # Model-specific errors - try next
            if "decommissioned" in error_str or "not found" in error_str:
                print(f"⚠️ Model {model} not available, trying next...")
                continue
            else:
                raise
    
    if resp is None:
        raise RuntimeError(
            f"❌ All Groq models failed.\n\n"
            f"**Last error:** {last_error}\n\n"
            f"**Solution:** Check GROQ_MODEL in .env or API status."
        )
    
    answer = resp.choices[0].message.content
    
    # Post-process to enforce readability (truncate overly long bullets)
    answer = improve_readability(answer)
    
    return answer


def ask_pdf(
    question: str,
    collections: List[str],
    top_k: int = MAX_CHUNKS,  # Use constant for consistency
    return_chunks: bool = False,
    document_context: Optional[List[Dict]] = None,
    chat_history: Optional[List[Dict]] = None
) -> Any:
    """
    Main RAG query function with query enhancement and comprehensive answers.
    
    Args:
        question: User's question
        collections: List of Qdrant collection names to search
        top_k: Number of chunks to retrieve (default: 5 for token safety)
        return_chunks: Whether to return chunk data for debugging
        document_context: List of document metadata for query enhancement
        chat_history: Previous Q&A pairs for context-aware enhancement
    
    Returns:
        Answer string, or tuple of (answer, chunks) if return_chunks=True
    """
    if not collections:
        raise ValueError("No collections provided for querying")
    
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    try:
        # Step 1: Enhance the query
        doc_context_str = get_document_context_string(document_context) if document_context else ""
        enhanced_query = enhance_query(
            original_query=question,
            document_context=doc_context_str,
            chat_history=chat_history
        )
        
        print(f"🔍 Enhanced query: {enhanced_query}")
        
        # Step 2: Retrieve relevant chunks
        chunks, raw_hits = retrieve_chunks(
            query=enhanced_query,
            collections=collections,
            top_k=top_k
        )
        
        if not chunks:
            no_info_msg = "I cannot find relevant information in the uploaded documents or websites to answer this question."
            if return_chunks:
                return no_info_msg, []
            return no_info_msg
        
        # Step 3: Build sources list
        sources_list = build_sources_list(chunks)
        
        # Step 4: Generate answer with strict token management
        answer = generate_answer(
            original_query=question,
            enhanced_query=enhanced_query,
            chunks=chunks,
            sources_list=sources_list
        )
        
        if return_chunks:
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'text': chunk['text'],
                    'score': chunk['score'],
                    'chunk_index': chunk['chunk_index'],
                    'source_name': chunk['source'],
                    'source_url': chunk.get('source_url', ''),
                    'timestamp': chunk.get('timestamp', 'N/A'),
                    'enhanced_query': enhanced_query
                })
            
            return answer, chunks_data
        
        return answer
        
    except Exception as e:
        raise RuntimeError(f"Error processing query: {str(e)}") from e