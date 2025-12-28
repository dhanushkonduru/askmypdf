import streamlit as st
import os
import re
import json
import hashlib
from datetime import datetime, timedelta
from urllib.parse import urlparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables early to prevent threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import model_cache (but don't load model yet - lazy loading)
import utils.model_cache

from pipelines.query_pdf import ask_pdf
from ingest.pdf_embedder import embed_pdf_to_qdrant
from ingest.web_scraper import crawl_urls
from ingest.web_embedder import embed_web_to_qdrant
from utils.error_handler import display_crawl_error
from utils.url_cache import URLCache

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize URL cache
url_cache = URLCache(cache_dir='.cache', ttl_hours=24)

st.set_page_config(
    page_title="📚 RAGChat - PDF & Web Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌟 Enhanced Custom CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app background with gradient */
    .stApp {
        font-family: 'Inter', 'Poppins', sans-serif;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 1px solid #334155;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Chat messages with glassmorphism */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 20px !important;
        margin: 12px 0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* User message */
    .user-msg {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 16px 16px 4px 16px;
        margin: 10px 0;
        margin-left: auto;
        margin-right: 0;
        max-width: 75%;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Assistant message */
    .bot-msg {
        background: rgba(51, 65, 85, 0.6);
        color: #f1f5f9;
        padding: 12px 18px;
        border-radius: 16px 16px 16px 4px;
        margin: 10px 0;
        margin-left: 0;
        margin-right: auto;
        max-width: 75%;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Links in sources */
    .stMarkdown a {
        color: #a78bfa !important;
        text-decoration: none;
        border-bottom: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stMarkdown a:hover {
        color: #c4b5fd !important;
        border-bottom: 2px solid #a78bfa;
    }
    
    /* Buttons with hover effects */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #334155;
        background: rgba(51, 65, 85, 0.5);
        color: #f1f5f9;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #a78bfa;
    }
    
    /* Source pills */
    .pdf-pill {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 8px 16px;
        margin: 6px 6px 6px 0;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #8b5cf6;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a78bfa;
    }
    
    /* Hide default elements */
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 🔧 Utility Functions
def clean_collection_name(filename: str) -> str:
    return filename.lower().replace(" ", "_").replace(".pdf", "")

def existing_qdrant_collections():
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        return [col.name for col in client.get_collections().collections]
    except Exception as e:
        st.error(f"❌ **Qdrant Connection Error:** {str(e)}")
        return []

def get_qdrant_client():
    """Get Qdrant client instance."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=qdrant_url, api_key=qdrant_key)

def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %I:%M %p')
    except:
        return timestamp_str

def get_document_icon(doc_type: str) -> str:
    """Get icon for document type."""
    icons = {'pdf': '📄', 'url': '🌐', 'website': '🌐'}
    return icons.get(doc_type, '📋')

def generate_query_suggestions(documents: list) -> list:
    """Generate contextual query suggestions based on loaded documents."""
    suggestions = []
    
    for doc in documents:
        doc_name = doc.get('name', '').lower()
        doc_type = doc.get('type', '')
        
        if doc_type == 'pdf':
            if 'resume' in doc_name:
                suggestions.extend([
                    "What are the key skills mentioned?",
                    "Summarize the work experience",
                    "What education background is listed?"
                ])
            elif 'report' in doc_name:
                suggestions.extend([
                    "What are the main findings?",
                    "Summarize the conclusions",
                    "What methodology was used?"
                ])
        elif doc_type in ['url', 'website']:
            if 'dinosaur' in doc_name.lower():
                suggestions.extend([
                    "What new dinosaurs were discovered?",
                    "When were these discoveries made?",
                    "What are the characteristics?"
                ])
            elif 'news' in doc.get('domain', '').lower():
                suggestions.append(f"Summarize the latest news from {doc.get('domain', 'this site')}")
    
    return list(set(suggestions))[:6]  # Limit to 6 unique suggestions

def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query keywords in text."""
    keywords = [w for w in query.lower().split() if len(w) > 3]
    for keyword in keywords:
        pattern = re.compile(f'({re.escape(keyword)})', re.IGNORECASE)
        text = pattern.sub(r'**\1**', text)
    return text

# Initialize Session State
if 'documents' not in st.session_state:
    st.session_state.documents = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'queries': 0,
        'documents': 0,
        'urls_crawled': 0,
        'pages_crawled': 0,
        'chunks_indexed': 0,
        'session_start': datetime.now(),
        'total_tokens': 0
    }

if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False

if 'retrieved_chunks' not in st.session_state:
    st.session_state.retrieved_chunks = []

# 📂 Header Section
col1, col2 = st.columns([4, 1], gap="small")
with col1:
    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <h1 style="margin-bottom: 0.25rem; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📚 RAGChat</h1>
        <p style="color: #94a3b8; font-size: 14px; margin-top: 0;">💬 Upload PDFs or crawl websites, then ask questions</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# Sidebar
with st.sidebar:
    # Upload PDF Section
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF file to embed and query",
        label_visibility="collapsed"
    )
    if uploaded_file:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        
        with st.spinner(f"📥 Embedding {uploaded_file.name}..."):
            try:
                collection_name = clean_collection_name(uploaded_file.name)
                embed_pdf_to_qdrant(save_path, collection_name=collection_name)
                
                # Add to documents
                doc_id = hashlib.md5(uploaded_file.name.encode()).hexdigest()
                doc_entry = {
                    'id': doc_id,
                    'name': uploaded_file.name,
                    'type': 'pdf',
                    'icon': '📄',
                    'timestamp': datetime.now().isoformat(),
                    'chunk_count': 0,  # Will be updated
                    'status': 'active',
                    'collection_name': collection_name
                }
                st.session_state.documents.append(doc_entry)
                st.session_state.stats['documents'] += 1
                
                st.success(f"✅ **{uploaded_file.name}** embedded successfully!")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error embedding PDF: {str(e)}")
    
    st.divider()
    
    # URL Ingestion Section
    st.header("🌐 Ingest Websites")
    url_input = st.text_area(
        "Enter URL(s)",
        placeholder="https://example.com\nOr multiple URLs separated by commas",
        help="Enter single URL or comma-separated list of URLs to crawl",
        label_visibility="collapsed",
        height=80
    )
    
    col_depth, col_pages = st.columns(2)
    with col_depth:
        crawl_depth = st.slider("Crawl Depth", 1, 5, 3, help="Number of levels to crawl")
    with col_pages:
        max_pages = st.slider("Max Pages", 5, 100, 50, step=5, help="Maximum pages to crawl")
    
    if st.button("🚀 Crawl & Embed URLs", use_container_width=True):
        if url_input and url_input.strip():
            urls_raw = url_input.replace('\n', ',').split(',')
            urls = [url.strip() for url in urls_raw if url.strip()]
            url_pattern = re.compile(r'^https?://.+')
            valid_urls = [url for url in urls if url_pattern.match(url)]
            
            if not valid_urls:
                st.error("❌ Please enter valid URLs (must start with http:// or https://)")
            else:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Use dict to track progress (avoids nonlocal issues)
                progress_state = {'pages_crawled': 0, 'pages_failed': 0, 'total_pages': 0}
                
                def progress_callback(url, status, depth):
                    progress_state['total_pages'] += 1
                    
                    if status == 'success':
                        progress_state['pages_crawled'] += 1
                    elif status == 'failed':
                        progress_state['pages_failed'] += 1
                    
                    progress = min(progress_state['total_pages'] / max_pages, 1.0)
                    progress_placeholder.progress(progress)
                    
                    status_placeholder.info(f"""
                    📊 **Crawling Progress**
                    - Current: {url[:50]}...
                    - Crawled: {progress_state['pages_crawled']} ✅
                    - Failed: {progress_state['pages_failed']} ❌
                    - Remaining: {max(0, max_pages - progress_state['total_pages'])}
                    """)
                
                try:
                    with st.spinner(f"🌐 Crawling {len(valid_urls)} URL(s)..."):
                        web_pages = crawl_urls(
                            valid_urls,
                            max_depth=crawl_depth,
                            max_pages=max_pages,
                            progress_callback=progress_callback
                        )
                        
                        if web_pages:
                            first_domain = urlparse(valid_urls[0]).netloc.replace('.', '_').replace('-', '_')
                            first_domain = re.sub(r'[^a-zA-Z0-9_]', '_', first_domain)
                            collection_name = f"web_{first_domain}"
                            
                            with st.spinner("📥 Embedding web content..."):
                                num_chunks = embed_web_to_qdrant(web_pages, collection_name=collection_name)
                                
                                # Add to documents
                                doc_id = hashlib.md5(valid_urls[0].encode()).hexdigest()
                                doc_entry = {
                                    'id': doc_id,
                                    'name': web_pages[0].get('title', first_domain),
                                    'type': 'website',
                                    'icon': '🌐',
                                    'timestamp': datetime.now().isoformat(),
                                    'chunk_count': num_chunks,
                                    'status': 'active',
                                    'url': valid_urls[0],
                                    'domain': urlparse(valid_urls[0]).netloc,
                                    'page_count': len(web_pages),
                                    'collection_name': collection_name
                                }
                                st.session_state.documents.append(doc_entry)
                                st.session_state.stats['urls_crawled'] += 1
                                st.session_state.stats['pages_crawled'] += len(web_pages)
                                st.session_state.stats['chunks_indexed'] += num_chunks
                                
                                progress_placeholder.empty()
                                status_placeholder.success(f"""
                                ✅ **Crawl Complete!**
                                - Successfully crawled: {len(web_pages)} pages
                                - Chunks indexed: {num_chunks}
                                """)
                                st.balloons()
                                st.rerun()
                        else:
                            st.warning("⚠️ No content extracted from URLs.")
                except Exception as e:
                    error_msg = str(e)
                    if "robots.txt" in error_msg.lower():
                        display_crawl_error('robots_blocked', valid_urls[0])
                    elif "timeout" in error_msg.lower():
                        display_crawl_error('timeout', valid_urls[0])
                    else:
                        st.error(f"❌ **Error:** {error_msg}")
        else:
            st.warning("⚠️ Please enter at least one URL")
    
    st.divider()
    
    # Document Management Panel
    st.header("📚 Your Documents")
    
    if not st.session_state.documents:
        st.info("📄 Upload a PDF or crawl a website to get started", icon="ℹ️")
    else:
        for doc in st.session_state.documents:
            with st.expander(f"{doc['icon']} {doc['name'][:30]}...", expanded=False):
                st.markdown(f"**Type:** {doc['type']}")
                st.markdown(f"**Added:** {format_timestamp(doc['timestamp'])}")
                st.markdown(f"**Chunks:** {doc.get('chunk_count', 'N/A')}")
                st.markdown(f"**Status:** {doc.get('status', 'active')}")
                
                if doc['type'] == 'website':
                    st.markdown(f"**URL:** [{doc.get('domain', 'N/A')}]({doc.get('url', '#')})")
                    st.markdown(f"**Pages:** {doc.get('page_count', 0)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Remove", key=f"remove_{doc['id']}", use_container_width=True):
                        try:
                            # Delete from Qdrant
                            client = get_qdrant_client()
                            collection_name = doc.get('collection_name')
                            if collection_name:
                                try:
                                    client.delete_collection(collection_name)
                                except:
                                    pass
                            
                            # Remove from session state
                            st.session_state.documents = [d for d in st.session_state.documents if d['id'] != doc['id']]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing document: {e}")
                
                with col2:
                    if doc['type'] == 'website':
                        if st.button("🔄 Refresh", key=f"refresh_{doc['id']}", use_container_width=True):
                            st.info("🔄 Refreshing... This will re-crawl the URL.")
                            # TODO: Implement refresh logic
                            st.rerun()
        
        if st.session_state.documents:
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
                if st.checkbox("Confirm deletion", key="confirm_delete"):
                    try:
                        client = get_qdrant_client()
                        collections = existing_qdrant_collections()
                        for col in collections:
                            try:
                                client.delete_collection(col)
                            except:
                                pass
                        st.session_state.documents = []
                        st.session_state.stats['documents'] = 0
                        st.success("All documents cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing documents: {e}")
    
    st.divider()
    
    # Document Selection
    collections = existing_qdrant_collections()
    if collections:
        selected_docs = st.multiselect(
            "Select documents to query:",
            collections,
            default=st.session_state.get('selected_pdfs', collections[:1] if collections else []),
            help="Choose documents to search through"
        )
        st.session_state.selected_pdfs = selected_docs
    
    st.divider()
    
    # Export Section
    st.markdown("### 💾 Export")
    if st.session_state.chat_history:
        def generate_markdown_export():
            md = f"""# RAGChat Chat Export
**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Documents:** {len(st.session_state.documents)}
**Messages:** {len(st.session_state.chat_history)}

---

"""
            for msg in st.session_state.chat_history:
                md += f"## 👤 You\n\n{msg['question']}\n\n---\n\n"
                md += f"## 🤖 Assistant\n\n{msg['answer']}\n\n---\n\n"
            
            md += "\n## 📚 Sources Used\n\n"
            for doc in st.session_state.documents:
                if doc['type'] == 'website':
                    md += f"- 🌐 [{doc['name']}]({doc.get('url', '#')}) - Crawled: {format_timestamp(doc['timestamp'])}\n"
                else:
                    md += f"- 📄 {doc['name']} - Added: {format_timestamp(doc['timestamp'])}\n"
            
            return md
        
        export_md = generate_markdown_export()
        st.download_button(
            "📥 Download as Markdown",
            export_md,
            f"ragchat_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            "text/markdown",
            use_container_width=True
        )
        
        json_export = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'messages': st.session_state.chat_history,
            'documents': st.session_state.documents
        }, indent=2)
        
        st.download_button(
            "📊 Download as JSON",
            json_export,
            f"ragchat_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )
    
    st.divider()
    
    # Session Statistics
    st.markdown("### 📊 Session Stats")
    stats = st.session_state.stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("❓ Queries", stats['queries'])
        st.metric("📚 Documents", len(st.session_state.documents))
    with col2:
        st.metric("🌐 URLs", stats['urls_crawled'])
        st.metric("📄 Pages", stats['pages_crawled'])
    
    session_time = datetime.now() - stats['session_start']
    st.caption(f"⏱️ Session Time: {str(session_time).split('.')[0]}")
    
    st.divider()
    
    # Debug Mode Toggle
    st.session_state.show_debug = st.checkbox("🐛 Show Debug Mode", value=st.session_state.show_debug)
    
    # Cache Management
    if st.button("🗑️ Clear URL Cache", use_container_width=True):
        url_cache.clear_all()
        st.success("Cache cleared!")

# Load embedded collections on startup
if 'collections_loaded' not in st.session_state:
    try:
        with st.spinner("Loading PDFs from uploads folder..."):
            # This would load existing PDFs - keeping existing logic
            st.session_state.collections_loaded = True
    except Exception as e:
        st.warning(f"⚠️ Could not load PDFs: {str(e)}")
        st.session_state.collections_loaded = True

# Main Content - Chat History
if st.session_state.chat_history:
    for idx, chat in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(
            f'<div class="user-msg"><strong>👤 You:</strong><br>{chat["question"]}</div>',
            unsafe_allow_html=True
        )
        
        # Assistant message
        st.markdown(
            f'<div class="bot-msg"><strong>🤖 Assistant:</strong><br>{chat["answer"]}</div>',
            unsafe_allow_html=True
        )
        
        # Enhanced Source Display
        if chat.get("sources"):
            sources_html = []
            for source in chat["sources"]:
                source_type = source.get("type", "pdf")
                icon = "🌐" if source_type == "website" else "📄"
                name = source.get("name", "Unknown")
                url = source.get("url", "")
                timestamp = source.get("timestamp", "")
                
                if url:
                    favicon_url = f"https://www.google.com/s2/favicons?domain={urlparse(url).netloc}"
                    source_display = f'<img src="{favicon_url}" width="16" height="16" style="vertical-align: middle; margin-right: 4px;"> {icon} <a href="{url}" target="_blank">{name}</a>'
                else:
                    source_display = f'{icon} {name}'
                
                if timestamp:
                    source_display += f' - Added: {format_timestamp(timestamp)}'
                
                sources_html.append(f'<span class="pdf-pill">{source_display}</span>')
            
            if sources_html:
                st.markdown(f'<div style="margin: 4px 0 12px 0; font-size: 11px;">📎 {" ".join(sources_html)}</div>', unsafe_allow_html=True)
        
        # Debug Mode - Retrieved Chunks Preview
        if st.session_state.show_debug and chat.get("chunks"):
            with st.expander("🔍 View Retrieved Context (Debug Mode)", expanded=False):
                for chunk_idx, chunk in enumerate(chat["chunks"], 1):
                    st.markdown(f"### Chunk {chunk_idx} - Relevance Score: {chunk.get('score', 0):.3f}")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if chunk.get('source_url'):
                            st.markdown(f"📍 **Source:** [{chunk.get('source_name', 'Unknown')}]({chunk.get('source_url')})")
                        else:
                            st.markdown(f"📍 **Source:** {chunk.get('source_name', 'Unknown')}")
                    with col2:
                        st.caption(f"Position: {chunk.get('chunk_index', 'N/A')}")
                    
                    highlighted = highlight_query_terms(chunk.get('text', ''), chat["question"])
                    st.markdown(highlighted)
                    
                    with st.expander("ℹ️ Chunk Metadata"):
                        st.json({
                            'embedding_distance': chunk.get('score', 0),
                            'timestamp': chunk.get('timestamp', 'N/A')
                        })
                    st.markdown("---")
        
        if idx < len(st.session_state.chat_history) - 1:
            st.markdown('<hr style="margin: 12px 0; border: none; border-top: 1px solid #334155;">', unsafe_allow_html=True)

# Smart Query Suggestions
if st.session_state.documents:
    st.markdown("#### 💡 Try asking:")
    suggestions = generate_query_suggestions(st.session_state.documents)
    
    if suggestions:
        col1, col2 = st.columns(2)
        for idx, suggestion in enumerate(suggestions):
            with col1 if idx % 2 == 0 else col2:
                if st.button(suggestion, key=f"suggest_{idx}", use_container_width=True):
                    st.session_state.current_query = suggestion
                    st.rerun()

# Question Input
st.markdown('<div style="margin: 0.5rem 0;"></div>', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col_q1, col_q2 = st.columns([6, 1], gap="small")
    with col_q1:
        user_query = st.text_input(
            "💬 Ask a question",
            placeholder="Type your question here...",
            value=st.session_state.get('current_query', ''),
            label_visibility="collapsed"
        )
    with col_q2:
        submitted = st.form_submit_button("🚀 Ask", use_container_width=True)
    
    if 'current_query' in st.session_state:
        del st.session_state.current_query

# Answer Logic
if submitted:
    selected_pdfs = st.session_state.get('selected_pdfs', [])
    
    if user_query.strip() and selected_pdfs:
        try:
            with st.spinner("🤔 Thinking... This may take a moment"):
                if st.session_state.show_debug:
                    answer, chunks = ask_pdf(user_query, collections=selected_pdfs, top_k=6, return_chunks=True)
                else:
                    answer = ask_pdf(user_query, collections=selected_pdfs, top_k=6, return_chunks=False)
                    chunks = []
            
            # Determine sources for display
            sources = []
            for coll in selected_pdfs:
                # Find matching document
                matching_doc = next((d for d in st.session_state.documents if d.get('collection_name') == coll), None)
                if matching_doc:
                    source_entry = {
                        "type": matching_doc['type'],
                        "name": matching_doc['name'],
                        "timestamp": matching_doc['timestamp']
                    }
                    if matching_doc['type'] == 'website':
                        source_entry['url'] = matching_doc.get('url', '')
                    sources.append(source_entry)
                else:
                    # Fallback
                    if coll.startswith("web_"):
                        sources.append({"type": "website", "name": coll.replace("web_", "").replace("_", ".")})
                    else:
                        sources.append({"type": "pdf", "name": coll})
            
            # Add to chat history
            chat_entry = {
                "question": user_query,
                "answer": answer,
                "sources": sources,
                "pdfs": selected_pdfs
            }
            if chunks:
                chat_entry["chunks"] = chunks
            st.session_state.chat_history.append(chat_entry)
            
            st.session_state.stats['queries'] += 1
            st.rerun()
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")
    else:
        if not user_query.strip():
            st.warning("⚠️ Please enter a question.")
        elif not selected_pdfs:
            st.warning("⚠️ Please select at least one document to query.")
