import streamlit as st
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables early to prevent threading issues
# Do this before any imports that might use threading
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import model_cache (but don't load model yet - lazy loading)
# This ensures threading environment variables are set before any model loading
import utils.model_cache

from pipelines.query_pdf import ask_pdf
from ingest.pdf_embedder import embed_pdf_to_qdrant

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="📚 PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌟 Premium Modern Style
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles - Clean Modern Design */
    .stApp {
        font-family: 'Inter', 'Poppins', sans-serif;
        background: #0f172a;
    }
    
    /* Main Container Background - Optimized */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1600px;
    }
    
    /* Sidebar Styling - Premium Design */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Sidebar Content Styling */
    [data-testid="stSidebar"] .element-container {
        padding: 0.5rem 0;
    }
    
    /* Sidebar Headers - Better Styling */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
    }
    
    /* Sidebar Text - Better Readability */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: #cbd5e1;
        font-size: 14px;
    }
    
    /* Sidebar Expander - Static */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: #1e293b;
        color: #f1f5f9;
        border-radius: 10px;
        padding: 10px 12px;
        border: 1px solid #334155;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar File Uploader - Static */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #1e293b;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* Sidebar Multiselect - Static */
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        color: #f1f5f9;
    }
    
    /* Sidebar Info Boxes */
    [data-testid="stSidebar"] .stInfo,
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stWarning,
    [data-testid="stSidebar"] .stError {
        background: #1e293b;
        border-radius: 10px;
        border-left: 4px solid;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    /* Sidebar Divider */
    [data-testid="stSidebar"] hr {
        border-color: #334155;
        margin: 1rem 0;
    }
    
    /* Sidebar Button */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem;
        font-weight: 600;
        font-size: 14px;
        margin: 0.25rem 0;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }
    
    /* Main Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    /* Header Styles - Compact */
    h1 {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.25rem;
        margin-top: 0;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    
    /* User Message Bubble - Compact - No Animation */
    .user-msg {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 16px 16px 4px 16px;
        margin: 10px 0;
        margin-left: auto;
        margin-right: 0;
        max-width: 75%;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        position: relative;
        font-size: 14px;
        line-height: 1.5;
    }
    
    .user-msg strong {
        font-weight: 600;
        opacity: 0.95;
        font-size: 13px;
    }
    
    /* Bot Message Bubble - Compact - No Animation */
    .bot-msg {
        background: #ffffff;
        color: #1e293b;
        padding: 12px 18px;
        border-radius: 16px 16px 16px 4px;
        margin: 10px 0;
        margin-left: 0;
        margin-right: auto;
        max-width: 75%;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        position: relative;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .bot-msg strong {
        color: #3b82f6;
        font-weight: 600;
    }
    
    .bot-msg::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 20px 0 0 20px;
    }
    
    /* PDF Pills - Static - No Animation */
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
        letter-spacing: 0.01em;
    }
    
    /* No animations - static display */
    
    /* Button Styles - Static - No Animation */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }
    
    /* Input Styles - Modern Inputs */
    .stTextInput > div > div > input {
        border-radius: 14px;
        border: 2px solid #334155;
        background: #1e293b;
        color: #f1f5f9;
        padding: 14px 18px;
        font-size: 15px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        background: #1e293b;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #64748b;
    }
    
    /* Select Box Styles */
    .stMultiSelect > div > div {
        border-radius: 14px;
        border: 2px solid #334155;
        background: #1e293b;
    }
    
    .stMultiSelect > div > div > div {
        background: #1e293b;
        color: #f1f5f9;
    }
    
    /* Card Styles */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .status-success {
        background: #c6f6d5;
        color: #22543d;
    }
    
    .status-error {
        background: #fed7d7;
        color: #742a2a;
    }
    
    .status-warning {
        background: #feebc8;
        color: #7c2d12;
    }
    
    /* Hide Streamlit default elements but keep sidebar toggle */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Ensure sidebar is visible and accessible */
    [data-testid="stSidebar"] {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Sidebar toggle button - make it prominent */
    [data-testid="stSidebarCollapseButton"],
    button[title="Close sidebar"],
    button[title="Open sidebar"] {
        visibility: visible !important;
        display: block !important;
        z-index: 999;
    }
    
    /* Force sidebar to be open and visible */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
    }
    
    /* Make sure sidebar content is visible */
    section[data-testid="stSidebar"] > div {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Chat Container - Optimized Space */
    .chat-container {
        max-height: calc(100vh - 300px);
        min-height: calc(100vh - 350px);
        overflow-y: auto;
        padding: 1.5rem;
        background: #1e293b;
        border-radius: 16px;
        margin: 0.5rem 0;
        border: 1px solid #334155;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Message Wrapper */
    .message-wrapper {
        display: flex;
        flex-direction: column;
        margin-bottom: 1.5rem;
    }
    
    .user-msg-wrapper {
        display: flex;
        justify-content: flex-end;
    }
    
    .bot-msg-wrapper {
        display: flex;
        justify-content: flex-start;
    }
    
    /* Scrollbar Styling - Modern */
    .chat-container::-webkit-scrollbar {
        width: 10px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #0f172a;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 10px;
        border: 2px solid #0f172a;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #2563eb 0%, #7c3aed 100%);
    }
    
    /* Sidebar Headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    
    /* Sidebar Text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] div {
        color: #cbd5e1;
    }
    
    /* Main Content Text */
    .main p, .main div, .main span {
        color: #f1f5f9;
    }
    
    /* Caption Styling */
    .stCaption {
        color: #94a3b8;
        font-size: 14px;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #1e293b;
        color: #f1f5f9;
        border-radius: 10px;
        padding: 12px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #1e293b;
        border: 2px dashed #475569;
        border-radius: 14px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: #1e293b;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: #1e293b;
        border-left: 4px solid #10b981;
        border-radius: 10px;
    }
    
    .stError {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        border-radius: 10px;
    }
    
    .stWarning {
        background: #1e293b;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
    }
    
    .stInfo {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border-color: #334155;
        margin: 1.5rem 0;
    }
    
    /* Selectbox Options */
    [data-baseweb="select"] {
        background: #1e293b;
        color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# 🔧 Utilities
def clean_collection_name(filename: str) -> str:
    return filename.lower().replace(" ", "_").replace(".pdf", "")

def existing_qdrant_collections():
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        return [col.name for col in client.get_collections().collections]
    except Exception as e:
        st.error(f"❌ **Qdrant Connection Error:** {str(e)}\n\n"
                "Please check your Qdrant configuration in the `.env` file:\n"
                "```\nQDRANT_URL=http://localhost:6333\nQDRANT_API_KEY=your_key_here\n```\n"
                "Make sure Qdrant is running if using local instance.")
        return []

def embed_all_pdfs_in_folder(folder_path=UPLOAD_DIR):
    try:
        collections = existing_qdrant_collections()
    except:
        collections = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            full_path = os.path.join(folder_path, file_name)
            collection_name = clean_collection_name(file_name)
            if collection_name not in collections:
                try:
                    st.toast(f"📥 Embedding {file_name}...")
                    embed_pdf_to_qdrant(full_path, collection_name=collection_name)
                    st.toast(f"✅ Embedded `{file_name}`")
                except Exception as e:
                    st.error(f"❌ Failed to embed {file_name}: {str(e)}")
                    print(f"Error embedding {file_name}: {str(e)}")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 📂 Header Section - Compact
col1, col2 = st.columns([4, 1], gap="small")
with col1:
    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <h1 style="margin-bottom: 0.25rem;">📚 Chat with Your PDFs</h1>
        <p style="color: #94a3b8; font-size: 14px; margin-top: 0;">💬 Upload PDFs, embed them, and ask questions</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# Sidebar - Upload and PDF Selection
with st.sidebar:
    # Upload Section
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
                embed_pdf_to_qdrant(save_path, collection_name=clean_collection_name(uploaded_file.name))
                st.success(f"✅ **{uploaded_file.name}** embedded successfully!")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error embedding PDF: {str(e)}")
    
    st.divider()
    
    # PDF Selection Section
    st.header("📑 Your PDFs")
    
    # Get collections
    collections = existing_qdrant_collections()
    
    if not collections:
        st.info("📄 Upload a PDF to get started", icon="ℹ️")
        st.session_state.selected_pdfs = []
    else:
        # Initialize selected_pdfs in session state if not exists
        if 'selected_pdfs' not in st.session_state:
            st.session_state.selected_pdfs = collections[:1] if collections else []
        
        # Multiselect for PDFs
        selected_pdfs = st.multiselect(
            "Select PDFs to query:",
            collections,
            default=st.session_state.selected_pdfs,
            help="Choose one or more PDFs to search through",
            label_visibility="collapsed"
        )
        
        # Update session state
        st.session_state.selected_pdfs = selected_pdfs
        
        st.markdown(f"""
        <div style="background: #1e293b; padding: 0.75rem; border-radius: 10px; margin-top: 0.75rem; border: 1px solid #334155; font-size: 12px;">
            <p style="color: #cbd5e1; margin: 0.25rem 0;">📊 {len(collections)} available</p>
            {f'<p style="color: #10b981; margin: 0.25rem 0;">✅ {len(selected_pdfs)} selected</p>' if selected_pdfs else '<p style="color: #64748b; margin: 0.25rem 0;">No selection</p>'}
        </div>
        """, unsafe_allow_html=True)

# 📌 Load embedded collections (only run if not already processed)
if 'collections_loaded' not in st.session_state:
    try:
        with st.spinner("Loading PDFs from uploads folder..."):
            embed_all_pdfs_in_folder()
            st.session_state.collections_loaded = True
            # Refresh collections after embedding
            collections = existing_qdrant_collections()
            if collections:
                st.success(f"✅ Loaded {len(collections)} PDF collection(s)")
    except Exception as e:
        st.warning(f"⚠️ Could not load PDFs: {str(e)}")
        st.session_state.collections_loaded = True  # Mark as loaded to prevent retry loop

# Main Content Area - Full Width for Chat
# Chat History Display - No Box Container
if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history):
            # User message wrapper
            st.markdown(
                f'<div class="message-wrapper user-msg-wrapper">'
                f'<div class="user-msg">'
                f'<strong>👤 You:</strong><br>{chat["question"]}'
                f'</div></div>',
                unsafe_allow_html=True
            )
            
            # Bot message wrapper
            st.markdown(
                f'<div class="message-wrapper bot-msg-wrapper">'
                f'<div class="bot-msg">'
                f'<strong>🤖 Assistant:</strong><br>{chat["answer"]}'
                f'</div></div>',
                unsafe_allow_html=True
            )
            
            # PDF sources - Compact
            if chat.get("pdfs"):
                pdf_tags = " ".join([f'<span class="pdf-pill">{pdf}</span>' for pdf in chat["pdfs"]])
                st.markdown(f'<div style="margin: 4px 0 12px 0; font-size: 11px; color: #64748b;">📎 {pdf_tags}</div>', unsafe_allow_html=True)
            
            # Only show divider if not last message
            if idx < len(st.session_state.chat_history) - 1:
                st.markdown('<hr style="margin: 12px 0; border: none; border-top: 1px solid #334155;">', unsafe_allow_html=True)

# 💬 Question Input - Compact
st.markdown('<div style="margin: 0.5rem 0;"></div>', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col_q1, col_q2 = st.columns([6, 1], gap="small")
    with col_q1:
        user_query = st.text_input(
            "💬 Ask a question",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
    with col_q2:
        submitted = st.form_submit_button("🚀 Ask", use_container_width=True)

# 🤖 Answer Logic
if submitted:
    # Get selected PDFs from sidebar (stored in session state)
    selected_pdfs = st.session_state.get('selected_pdfs', [])
    
    if user_query.strip() and selected_pdfs:
        try:
            with st.spinner("🤔 Thinking... This may take a moment"):
                answer = ask_pdf(user_query, collections=selected_pdfs, top_k=6)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_query,
                "answer": answer,
                "pdfs": selected_pdfs
            })
            
            st.rerun()
        except ValueError as e:
            if "GROQ_API_KEY" in str(e):
                st.error("""
                <div class="info-card">
                    <h3>🔑 API Key Missing</h3>
                    <p>Please create a `.env` file in the `rag_chatbot` directory with:</p>
                    <pre>GROQ_API_KEY=your_api_key_here</pre>
                    <p>Get your API key from: <a href="https://console.groq.com/keys" target="_blank">https://console.groq.com/keys</a></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ **Error:** {str(e)}")
        except Exception as e:
            error_msg = str(e)
            # Check for common errors and provide helpful messages
            if "GROQ_API_KEY" in error_msg or "api_key" in error_msg.lower():
                st.error("""
                <div class="info-card">
                    <h3>🔑 API Configuration Error</h3>
                    <p>Please check your `.env` file and ensure `GROQ_API_KEY` is set correctly.</p>
                    <p><strong>Error:</strong> {}</p>
                </div>
                """.format(error_msg), unsafe_allow_html=True)
            elif "QDRANT" in error_msg.upper() or "qdrant" in error_msg.lower():
                st.error("""
                <div class="info-card">
                    <h3>🗄️ Qdrant Connection Error</h3>
                    <p>Please check your Qdrant connection settings in the `.env` file.</p>
                    <p><strong>Error:</strong> {}</p>
                </div>
                """.format(error_msg), unsafe_allow_html=True)
            elif "model" in error_msg.lower() and "decommissioned" in error_msg.lower():
                st.warning("""
                <div class="info-card">
                    <h3>🤖 Model Updated</h3>
                    <p>The model has been automatically updated. Please try your question again.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("""
                <div class="info-card">
                    <h3>❌ Error Occurred</h3>
                    <p><strong>Details:</strong> {}</p>
                    <p>If this persists, check your `.env` file configuration and ensure all services are running.</p>
                </div>
                """.format(error_msg), unsafe_allow_html=True)
    else:
        if not user_query.strip():
            st.warning("⚠️ Please enter a question.")
        elif not selected_pdfs:
            st.warning("⚠️ Please select at least one PDF to query.")
        else:
            st.warning("⚠️ Please enter a question and select at least one PDF.")
