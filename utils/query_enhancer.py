"""
Query Enhancement Module using Groq API.

This module provides intelligent query rewriting to improve RAG retrieval results by:
1. Making implicit references explicit
2. Adding relevant technical context
3. Expanding abbreviations
4. Optimizing for semantic search
"""

import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
from utils.groq_client import get_groq_client

load_dotenv()


def enhance_query(
    original_query: str,
    document_context: str = "",
    chat_history: Optional[List[Dict]] = None
) -> str:
    """
    Enhance a user query to be more specific and retrieval-friendly.
    
    Args:
        original_query: The user's original question
        document_context: Description of available documents (e.g., "GitHub repo about solar panels")
        chat_history: List of previous Q&A turns for context
    
    Returns:
        Enhanced query string, or original query if enhancement fails
    """
    try:
        groq = get_groq_client()
        
        # Build conversation history context
        turn_number = len(chat_history) + 1 if chat_history else 1
        last_exchange = ""
        
        if chat_history and len(chat_history) > 0:
            last_turn = chat_history[-1]
            last_q = last_turn.get('question', '')
            last_a = last_turn.get('answer', '')[:200] + "..." if len(last_turn.get('answer', '')) > 200 else last_turn.get('answer', '')
            last_exchange = f"User: {last_q}\nAssistant: {last_a}"
        
        # Build the enhancement prompt (PROMPT 1: Query Enhancement with intent preservation)
        document_list = document_context if document_context else "No documents specified"
        prompt = f"""You are a query optimizer for a strict RAG document system.

CONTEXT:
- User has uploaded: {document_list}
- Current conversation turn: {turn_number}
- Previous context: {last_exchange if last_exchange else "None"}

USER'S QUERY: "{original_query}"

YOUR TASK:
Transform this query into a retrieval-optimized search query while preserving intent:

1. If query asks about a PERSON (who is X, tell me about X, X's profile, biography of X):
   - Enhance to: "Find biographical information, professional background, and personal description of [person name]"
   - Focus on: bio text, personal description, background information, achievements WITH details
   - DO NOT target platform features, repository lists, or generic capabilities

2. If query asks about a PROJECT/TECH/CONCEPT:
   - Enhance with technical terms and specifics
   - Replace pronouns with explicit project/technology names
   - Add relevant technical keywords from the domain
   - Expand acronyms and add synonyms

3. If query asks about PLATFORM FEATURES:
   - Keep focused on platform features and capabilities

OUTPUT FORMAT (CRITICAL):
Return ONLY the enhanced query as plain text. No explanations. No preamble.

EXAMPLES:
Input: "tell me about dhanush" (person query)
Output: "Find biographical information, professional background, and personal description of dhanush"

Input: "what does it help for" (project query)
Output: "What is the purpose and functionality of the sun-tracking solar panel system? What problems does this IoT device solve?"

Input: "how does it work" (technical query)
Output: "How does the Smart IoT device operate? Explain the technical architecture and sensor mechanisms."

Enhanced query:"""

        # Use Groq API with model selection and fallbacks
        user_model = os.getenv("GROQ_MODEL")
        fallback_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "llama-3.1-70b-versatile"
        ]
        
        model_name = user_model if user_model else fallback_models[0]
        models_to_try = [model_name] + [m for m in fallback_models if m != model_name]
        
        resp = None
        for model in models_to_try:
            try:
                resp = groq.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for authentication errors (401) - API key issue
                if "401" in str(e) or "invalid_api_key" in error_str or "invalid api key" in error_str or "unauthorized" in error_str:
                    # For query enhancement, just return original query instead of raising
                    # The main query will catch and display the error properly
                    return original_query
                
                # Check for model-specific errors - try next model
                if "decommissioned" in error_str or "not found" in error_str or ("invalid" in error_str and "model" in error_str):
                    continue
                else:
                    raise
        
        if resp is None:
            # Fallback: return original query if all models failed
            return original_query
        
        enhanced = resp.choices[0].message.content.strip()
        
        # Clean up any prefixes the model might have added
        prefixes_to_remove = [
            "Enhanced query:", "Enhanced Query:", 
            "Here's the enhanced query:", "The enhanced query is:",
            '"', "'"
        ]
        for prefix in prefixes_to_remove:
            if enhanced.startswith(prefix):
                enhanced = enhanced[len(prefix):].strip()
            if enhanced.endswith('"') or enhanced.endswith("'"):
                enhanced = enhanced[:-1].strip()
        
        return enhanced if enhanced else original_query
        
    except Exception as e:
        # Log error but don't fail - return original query
        print(f"Query enhancement failed: {e}")
        return original_query


def get_document_context_string(documents: List[Dict]) -> str:
    """
    Build a context string describing available documents.
    
    Args:
        documents: List of document metadata dicts from session state
    
    Returns:
        Human-readable description of available documents
    """
    if not documents:
        return ""
    
    context_parts = []
    for doc in documents:
        doc_type = doc.get('type', 'unknown')
        name = doc.get('name', 'Unknown')
        
        if doc_type == 'website':
            url = doc.get('url', '')
            context_parts.append(f"- Website: {name} ({url})")
        elif doc_type == 'pdf':
            context_parts.append(f"- PDF Document: {name}")
        else:
            context_parts.append(f"- Document: {name}")
    
    return "\n".join(context_parts)

