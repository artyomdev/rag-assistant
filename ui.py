# ui.py
"""
Streamlit UI for the RAG system with support for clarification questions.

Features:
- Display of clarification options (disambiguation).
- Structured sources with clickable links.
- Debug panel with entity and intent information.
- Loading indicators while requests are processed.
"""
import streamlit as st
import requests
import time
import os
API_URL = os.environ.get("API_URL", "http://localhost:9004")
st.set_page_config(
    page_title="RAG Assistant v2",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🤖 Knowledge Base Assistant")

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None
if "original_query" not in st.session_state:
    st.session_state.original_query = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False


def display_sources(sources: list):
    """Render structured sources in a compact grid."""
    if not sources:
        return
    
    st.markdown("---")
    st.markdown("**📚 Sources:**")
    
    cols = st.columns(min(len(sources), 3))
    for i, source in enumerate(sources):
        with cols[i % 3]:
            name = source.get('name', 'Document')
            link = source.get('link', '')
            category = source.get('category', '')
            
            if link:
                st.markdown(f"[📄 {name}]({API_URL}{link})")
                if category:
                    st.caption(f"Category: {category}")
            else:
                st.markdown(f"📄 {name}")
                if category:
                    st.caption(f"Category: {category}")


def display_clarification(clarification: dict):
    """Render clarification options, grouped by category when available."""
    question = clarification.get('question', 'Please clarify your question:')
    options = clarification.get('options', [])
    reason = clarification.get('reason', '')
    
    if st.session_state.is_loading:
        st.info("⏳ Processing your choice...")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        return
    
    st.warning(f"❓ {question}")
    
    by_category = {}
    no_category = []
    for opt in options:
        cat = opt.get('category') or ''
        if cat:
            by_category.setdefault(cat, []).append(opt)
        else:
            no_category.append(opt)
    
    def render_option(option: dict):
        option_id = option.get('id', '')
        label = option.get('label', '')
        description = option.get('description', '')
        substituted_query = option.get('substituted_query')
        button_label = f"**{label}**"
        if description:
            button_label += f"\n\n_{description[:100]}..._" if len(description) > 100 else f"\n\n_{description}_"
        if st.button(button_label, key=f"clarify_{option_id}", use_container_width=True):
            st.session_state.is_loading = True
            source_name = description if option_id.startswith('topic_') and option_id != 'topic_show_all' else None
            handle_clarification_choice(option_id, source_name=source_name, substituted_query=substituted_query)
    
    for cat in sorted(by_category.keys(), key=str.lower):
        st.markdown(f"**📁 {cat}**")
        for option in by_category[cat]:
            render_option(option)
        st.markdown("")
    
    for option in no_category:
        render_option(option)


def handle_clarification_choice(choice_id: str, source_name: str = None, substituted_query: str = None):
    """Handle user clarification choice (including catalog options)."""
    original_query = st.session_state.original_query
    
    if not original_query:
        st.session_state.is_loading = False
        st.error("Error: original query was not found")
        return
    
    try:
        payload = {
            "choice_id": choice_id,
            "original_query": original_query
        }
        if source_name:
            payload["source_name"] = source_name
        if substituted_query:
            payload["substituted_query"] = substituted_query
        
        with st.spinner("🔍 Generating answer..."):
            response = requests.post(
                f"{API_URL}/clarify",
                json=payload,
                timeout=120
            )
        
        if response.status_code == 200:
            data = response.json()
            
            st.session_state.pending_clarification = None
            st.session_state.original_query = None
            st.session_state.is_loading = False
            
            if data.get('answer'):
                st.session_state.history.append({
                    "role": "assistant",
                    "content": data['answer'],
                    "sources": data.get('sources', []),
                    "debug": data.get('debug', {})
                })
            
            st.rerun()
        else:
            st.session_state.is_loading = False
            st.error(f"Server error: {response.text}")
    except Exception as e:
        st.session_state.is_loading = False
        st.error(f"Connection error: {e}")


def process_query(query: str):
    """Send user query to the backend API."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": query},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('needs_clarification'):
                st.session_state.pending_clarification = data.get('clarification', {})
                st.session_state.original_query = query
                return None
            else:
                return data
        else:
            st.error(f"Server error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


# Read input immediately: if user sends a message we reset pending clarification
prompt = st.chat_input("Your question...")
if prompt:
    if st.session_state.pending_clarification:
        st.session_state.pending_clarification = None
        st.session_state.original_query = None
    st.session_state.pending_prompt = prompt


# Render chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant":
            if msg.get("sources"):
                display_sources(msg["sources"])


# Show clarification only when we are not processing a new message in this run
if st.session_state.pending_clarification and not st.session_state.pending_prompt:
    with st.chat_message("assistant"):
        display_clarification(st.session_state.pending_clarification)


# Process pending message (clarification or new question)
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    
    st.session_state.history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your request..."):
            result = process_query(prompt)
        
        if result:
            st.markdown(result.get('answer', ''))
            
            sources = result.get('sources', [])
            display_sources(sources)
            
            with st.expander("🛠 Technical details"):
                debug = result.get('debug', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Query analysis:**")
                    st.write(f"- Canonical: `{debug.get('canonical_query', 'N/A')}`")
                    st.write(f"- Intent: `{debug.get('intent', 'N/A')}`")
                    st.write(f"- Confidence: `{debug.get('confidence', 0):.2f}`")
                
                with col2:
                    st.markdown("**Retrieval:**")
                    st.write(f"- Top Score: `{debug.get('top_score', 'N/A')}`")
                    entity_ids = debug.get('entity_ids', [])
                    if entity_ids:
                        st.write(f"- Entity IDs: `{', '.join(entity_ids[:3])}`")
                
                context = result.get('context', '')
                if context:
                    st.markdown("**Context used:**")
                    st.text_area("Context", value=context, height=200, disabled=True)
            
            st.session_state.history.append({
                "role": "assistant",
                "content": result.get('answer', ''),
                "sources": sources,
                "debug": debug
            })
        
        elif st.session_state.pending_clarification:
            display_clarification(st.session_state.pending_clarification)


# Sidebar with system information
with st.sidebar:
    st.markdown("### 📊 System status")
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"✅ API v{health.get('version', '?')} is online")
    except:
        st.error("❌ API is not reachable")
    
    if st.button("📋 Show entities"):
        try:
            entities = requests.get(f"{API_URL}/entities", timeout=10).json()
            st.write(f"Total entities: {entities.get('count', 0)}")
            
            for e in entities.get('entities', [])[:10]:
                st.markdown(f"- **{e['label']}** ({e['type']}) - {e['frequency']} mentions")
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    
    if st.button("🗑 Clear history"):
        st.session_state.history = []
        st.session_state.pending_clarification = None
        st.session_state.original_query = None
        st.session_state.pending_prompt = None
        st.session_state.is_loading = False
        st.rerun()
