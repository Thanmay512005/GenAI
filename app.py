"""TravelMate Karnataka — Streamlit RAG travel assistant."""
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from document_processing import (
    DEFAULT_FAQ_PATH, FAQEntry, add_single_faq_entry, delete_single_faq_entry,
    load_faq_entries, save_faq_entries, update_single_faq_entry,
)
from notice_manager import (
    add_notice, delete_notice, get_active_notices, get_pinned_notices,
    load_notices, update_notice,
)
from question_manager import (
    delete_submitted_question, dismiss_question, get_pending_questions,
    load_submitted_questions, mark_question_answered, submit_question,
)
from rag_pipeline import (
    API_KEY_ERROR, GeminiResponseError, MissingAPIKeyError, RetrievalError,
    answer_question,
)
from vector_store import rebuild_index, search_faq

load_dotenv()
st.set_page_config(page_title="TravelMate Karnataka", page_icon="🧭", layout="wide")
st.markdown("""
<style>
:root { --ink:#15243b; --muted:#60718b; --accent:#0d766e; --gold:#d7a84b; }
.block-container {max-width: 1240px; padding-top: 1.35rem; padding-bottom: 3rem;}
/* Hampi heritage backdrop; the dark overlay keeps text readable. */
[data-testid="stAppViewContainer"] {
  background-image: linear-gradient(115deg, rgba(8,20,31,.91), rgba(13,42,49,.83)),
    url("https://upload.wikimedia.org/wikipedia/commons/a/a2/Stone_Chariot_at_Vittala_Temple_Complex%2C_Hampi.jpg");
  background-size: cover; background-position: center 34%; background-attachment: fixed;
}
[data-testid="stHeader"] {background: rgba(7,20,30,.48);}
[data-testid="stSidebar"] {background:rgba(8,25,35,.94); border-right:1px solid rgba(215,168,75,.28);}
[data-testid="stSidebar"] * {color:#f5f2e9;}
.block-container, [data-testid="stMain"] {color:#f7f4ec;}
.hero {padding:2.15rem 2.35rem;border:1px solid rgba(236,198,119,.42);border-radius:24px;
  background:linear-gradient(100deg,rgba(7,25,37,.94),rgba(12,67,62,.78));
  box-shadow:0 18px 55px rgba(0,0,0,.28);color:white;margin:0 0 1.15rem 0;}
.hero h1 {font-size:2.55rem;margin:0 0 .45rem 0;color:#fff;letter-spacing:.2px}
.hero p {color:#f3e6c9;font-size:1.04rem;margin:0}
.hero .kannada {font-size:1.08rem;color:#f2c76f;margin-top:.85rem;letter-spacing:.3px}
.karnataka-ribbon {display:flex;align-items:center;gap:.65rem;color:#f5d58d;font-weight:650;margin:.2rem 0 1rem 0;}
.gberunda {width:43px;height:36px;flex:none;filter:drop-shadow(0 2px 5px rgba(0,0,0,.35));}
.small-muted {color:#d8e1df;font-size:.9rem}
.source-card {padding:.8rem 1rem;border:1px solid rgba(255,255,255,.18);border-radius:12px;margin:.5rem 0;background:rgba(10,28,38,.72)}
.notice-card {padding:1rem;border:1px solid rgba(226,232,240,.25);border-left:4px solid #d7a84b;border-radius:12px;margin:.6rem 0;background:rgba(10,28,38,.76)}
div[data-testid="stChatMessage"] {border-radius:14px;background:rgba(8,26,37,.72);border:1px solid rgba(255,255,255,.12)}
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {background:rgba(255,255,255,.96)!important;color:#172536!important;border-radius:10px!important;}
.stButton button {border-radius:999px;border:1px solid rgba(231,190,108,.55);background:rgba(12,53,54,.88);color:#fff;font-weight:600;transition:.18s ease;}
.stButton button:hover {border-color:#f0c66f;background:#176c62;color:#fff;transform:translateY(-1px);}
[data-testid="stTabs"] button {color:#f4e9d1!important;}
[data-testid="stExpander"] {background:rgba(10,28,38,.68);border:1px solid rgba(255,255,255,.17);border-radius:12px;}
hr {border-color:rgba(255,255,255,.2)!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading semantic embedding model…")
def warmup_index():
    return rebuild_index()

def refresh_index():
    warmup_index.clear()
    return warmup_index()

def render_sources(sources):
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source.question} · {source.similarity:.0%} match"):
            st.caption(f"{source.source} · {source.category}")
            st.write(source.content)

def chat_tab():
    st.subheader("Ask the Karnataka travel assistant")
    st.caption("Answers are generated using retrieved TravelMate knowledge-base passages (RAG). Sources are shown with every grounded answer.")
    st.markdown("**✨ Try asking**")
    suggested = ["Plan a 2-day heritage trip to Hampi", "Best waterfalls and hill stations in Karnataka?", "Plan a 3-day Mysuru–Coorg trip", "What are famous foods in North Karnataka?"]
    suggestion_cols = st.columns(2)
    for i, question in enumerate(suggested):
        if suggestion_cols[i % 2].button(question, key=f"chat_suggest_{i}", use_container_width=True):
            st.session_state["suggested_prompt"] = question
            st.rerun()
    if "messages" not in st.session_state: st.session_state.messages=[]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"): render_sources(m["sources"])
    prompt = st.session_state.pop("suggested_prompt", None)
    typed_prompt = st.chat_input("Ask about Karnataka destinations, itineraries, food, travel planning…")
    if typed_prompt:
        prompt = typed_prompt
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the travel knowledge base and generating an answer…"):
                    result=answer_question(prompt,allow_extractive_fallback=True)
                st.markdown(result.answer)
                if result.is_extractive_fallback: st.info("Showing a direct retrieved answer because Gemini generation was unavailable.")
                elif result.model_used: st.caption(f"Generated with {result.model_used} · grounded in retrieved sources")
                render_sources(result.sources)
                st.session_state.messages.append({"role":"assistant","content":result.answer,"sources":result.sources})
            except MissingAPIKeyError:
                msg=f"🔑 {API_KEY_ERROR} Create a `.env` file using `.env.example`, then restart the app."
                st.warning(msg); st.session_state.messages.append({"role":"assistant","content":msg})
            except RetrievalError as e:
                st.error(str(e))
            except GeminiResponseError as e:
                st.error(f"Gemini is unavailable: {e.details}")
                render_sources(e.sources)
                st.info("Check your API key, model quota, and internet connection.")
    c1,c2=st.columns([1,5])
    with c1:
        if st.button("Clear chat"): st.session_state.messages=[]; st.rerun()

def load_district_directory():
    """Load the curated 31-district directory used by Explore and the RAG corpus."""
    import json
    path = Path(__file__).parent / "data" / "districts.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("districts", [])
    except (OSError, json.JSONDecodeError):
        return []

def explore_tab():
    st.subheader("Explore all 31 districts of Karnataka")
    st.caption("Browse district highlights, major attractions and suggested trip durations. Select a district to view detailed travel notes.")
    districts = load_district_directory()
    if not districts:
        st.warning("District guide could not be loaded. Check data/districts.json.")
        return
    st.metric("Districts covered", len(districts))
    search = st.text_input("Find a district or attraction", placeholder="e.g. Hampi, waterfalls, beaches, Bidar…")
    regions = ["All regions"] + sorted({d.get("region", "Other") for d in districts})
    selected_region = st.selectbox("Filter by revenue division", regions)
    filtered = [d for d in districts if (selected_region == "All regions" or d.get("region") == selected_region)]
    if search.strip():
        term = search.casefold()
        filtered = [d for d in filtered if term in d.get("district", "").casefold() or term in d.get("theme", "").casefold() or any(term in p.get("name", "").casefold() or term in p.get("detail", "").casefold() for p in d.get("destinations", []))]
    st.caption(f"Showing {len(filtered)} of {len(districts)} districts")
    names = [d["district"] for d in filtered]
    if not names:
        st.info("No matching district or attraction. Try a broader search.")
        return
    selected_name = st.selectbox("Choose a district", names)
    district = next(d for d in filtered if d["district"] == selected_name)
    st.markdown(f"### {district['district']}")
    st.caption(f"{district['region']} · {district['theme']}")
    st.write(district["overview"])
    st.markdown(f"**Best for:** {district['best_for']}  ")
    st.markdown(f"**Suggested duration:** {district['suggested_days']}")
    st.markdown("#### Main destinations")
    for place in district.get("destinations", []):
        with st.expander(place["name"]):
            st.write(place["detail"])
    st.markdown("#### Suggested trip planning")
    st.write(district["planning"])
    st.info("Travel details are curated general guidance, not live feeds. Confirm current opening hours, permits, road conditions, weather, transport and safety advisories with official providers before departure.")

def search_tab():
    st.subheader("Semantic travel knowledge search")
    st.caption("Search across the Karnataka district guides, destination notes and travel FAQs.")
    examples = ["Hampi heritage monuments", "Coorg family itinerary", "Karnataka waterfalls", "coastal Karnataka beaches", "Mysuru palace and nearby places", "North Karnataka food"]
    st.markdown("**Popular searches**")
    cols = st.columns(3)
    for i, example in enumerate(examples):
        if cols[i % 3].button(example, key=f"search_suggest_{i}", use_container_width=True):
            st.session_state["search_query"] = example
            st.session_state["search_requested"] = True
    query=st.text_input("Search the TravelMate knowledge base",placeholder="e.g. family trip to Mysuru in December", key="search_query")
    top_k=st.slider("Number of passages",1,8,4)
    do_search = st.button("Search knowledge base",type="primary",disabled=not query.strip())
    if do_search or st.session_state.pop("search_requested", False):
        try:
            with st.spinner("Finding semantically similar travel passages…"):
                results=search_faq(query,top_k=top_k)
            if not results: st.warning("No relevant passages found.")
            for i,r in enumerate(results,1):
                st.markdown(f"**{i}. {r.question}**")
                st.caption(f"{r.source} · {r.category} · similarity {r.similarity:.1%}")
                st.write(r.content)
                st.progress(max(0,min(1,r.similarity)))
        except Exception as e: st.error(f"Search failed: {e}")

def updates_tab():
    st.subheader("Travel updates and safety reminders")
    notices=get_active_notices()
    pinned=get_pinned_notices()
    if pinned:
        st.markdown("#### Pinned reminders")
        for n in pinned:
            st.markdown(f'<div class="notice-card"><b>📌 {n.title}</b><br>{n.content}</div>',unsafe_allow_html=True)
    st.markdown("#### All updates")
    if not notices: st.info("No active updates.")
    for n in notices:
        with st.expander(f"{n.title} · {n.category}"):
            st.write(n.content); st.caption(f"Posted {n.posted_date} · {n.priority} · {n.target_audience}")

def ask_tab():
    st.subheader("Ask a question for the TravelMate knowledge base")
    with st.form("question_form"):
        question=st.text_area("Your question",placeholder="What is the best way to plan a 4-day trip from Mysuru to Coorg?")
        name=st.text_input("Name (optional)")
        category=st.selectbox("Category",["Trip planning","Destinations","Transport","Food","Safety","Other"])
        sent=st.form_submit_button("Submit question",type="primary")
    if sent:
        if len(question.strip())<8: st.error("Please enter a little more detail.")
        else:
            submit_question(question.strip(),submitted_by=name.strip() or "Traveler",category=category)
            st.success("Your question has been submitted for review.")

def admin_tab():
    st.subheader("TravelMate administration")
    if "admin_auth" not in st.session_state: st.session_state.admin_auth=False
    if not st.session_state.admin_auth:
        with st.form("admin_login"):
            password=st.text_input("Admin password",type="password")
            login=st.form_submit_button("Sign in")
        if login:
            expected=os.getenv("ADMIN_PASSWORD","")
            if expected and password==expected: st.session_state.admin_auth=True; st.rerun()
            st.error("Incorrect password or ADMIN_PASSWORD is not configured in .env.")
        return
    if st.button("Sign out"): st.session_state.admin_auth=False; st.rerun()
    faq_col,notice_col,question_col=st.tabs(["📚 Knowledge base","📢 Notices","❓ Traveler questions"])
    with faq_col:
        entries=load_faq_entries()
        st.write(f"**{len(entries)} knowledge entries**")
        with st.form("add_faq"):
            q=st.text_input("Question"); a=st.text_area("Answer"); cat=st.text_input("Category",value="Trip planning")
            if st.form_submit_button("Add knowledge entry"):
                if q.strip() and a.strip():
                    add_single_faq_entry(q,a,cat); refresh_index(); st.success("Added and re-indexed."); st.rerun()
                else: st.error("Question and answer are required.")
        for i,e in enumerate(entries):
            with st.expander(f"{i+1}. {e.question}"):
                nq=st.text_input("Question",e.question,key=f"q{i}")
                na=st.text_area("Answer",e.answer,key=f"a{i}")
                if st.button("Save changes",key=f"save{i}"):
                    update_single_faq_entry(i,nq,na,e.category); refresh_index(); st.success("Updated."); st.rerun()
                if st.button("Delete",key=f"del{i}"):
                    delete_single_faq_entry(i); refresh_index(); st.warning("Deleted."); st.rerun()
        if st.button("Rebuild semantic index"): 
            with st.spinner("Rebuilding embeddings and ChromaDB index…"): count=refresh_index()
            st.success(f"Indexed {count} passages.")
    with notice_col:
        with st.form("new_notice"):
            title=st.text_input("Notice title"); content=st.text_area("Details")
            category=st.selectbox("Category",["Travel Safety","Transport","Planning","Destination Update","Responsible Travel"])
            priority=st.selectbox("Priority",["Normal","Important","Urgent"])
            pinned=st.checkbox("Pin this notice")
            if st.form_submit_button("Publish notice"):
                if title.strip() and content.strip():
                    add_notice(title,content,category,priority,pinned); refresh_index(); st.success("Published."); st.rerun()
                else: st.error("Title and details are required.")
        for n in load_notices():
            with st.expander(n.title):
                st.write(n.content); st.caption(f"{n.category} · {n.priority}")
                c1,c2=st.columns(2)
                with c1:
                    if st.button("Deactivate" if n.is_active else "Activate",key=f"active{n.id}"):
                        update_notice(n.id,is_active=not n.is_active); refresh_index(); st.rerun()
                with c2:
                    if st.button("Delete notice",key=f"ndel{n.id}"):
                        delete_notice(n.id); refresh_index(); st.rerun()
    with question_col:
        questions=load_submitted_questions()
        if not questions: st.info("No submitted questions.")
        for q in questions:
            with st.expander(f"{q.status}: {q.question}"):
                st.caption(f"Submitted by {q.submitted_by} · {q.category} · {q.submitted_date}")
                if q.answer: st.write(q.answer)
                answer=st.text_area("Response",key=f"resp{q.id}")
                c1,c2,c3=st.columns(3)
                with c1:
                    if st.button("Mark answered",key=f"ans{q.id}") and answer.strip():
                        mark_question_answered(q.id,answer); st.success("Answered."); st.rerun()
                with c2:
                    if st.button("Dismiss",key=f"dis{q.id}"): dismiss_question(q.id); st.rerun()
                with c3:
                    if st.button("Delete",key=f"qdel{q.id}"): delete_submitted_question(q.id); st.rerun()

st.markdown('''<div class="hero"><h1>🧭 TravelMate Karnataka</h1><p>Discover the land of timeless temples, royal cities, Western Ghats and the coast — powered by grounded RAG.</p><div class="kannada">ಕರ್ನಾಟಕದ ಹೆಮ್ಮೆ · ನಮ್ಮ ನಾಡು, ನಮ್ಮ ಸಂಸ್ಕೃತಿ, ನಮ್ಮ ಪರಂಪರೆ</div></div>
<div class="karnataka-ribbon"><svg class="gberunda" viewBox="0 0 120 90" role="img" aria-label="Gandaberunda-inspired heritage emblem"><path fill="#d7a84b" d="M58 42C43 15 16 10 4 22c13 1 17 8 19 16C8 32 2 43 8 53c12-8 23-7 34 1-7 8-7 17-1 25 9-4 15-13 19-24 4 11 10 20 19 24 6-8 6-17-1-25 11-8 22-9 34-1 6-10 0-21-15-15 2-8 6-15 19-16-12-12-39-7-54 20l-6 9z"/><path fill="#f4d28b" d="M24 22l-12-9 17 2 9 9M96 22l12-9-17 2-9 9"/><circle cx="31" cy="32" r="2.7" fill="#142c35"/><circle cx="89" cy="32" r="2.7" fill="#142c35"/><path d="M51 48l9 10 9-10-9-5z" fill="#a63d2f"/></svg><span>ಕನ್ನಡ ನಾಡು · ಕನ್ನಡ ನುಡಿ · ಕನ್ನಡ ಸಂಸ್ಕೃತಿ</span></div>''',unsafe_allow_html=True)
st.caption("Heritage background: Stone Chariot, Vittala Temple Complex, Hampi. Photo: Prakruth Gowda · CC BY-SA 4.0 · Wikimedia Commons. Background image is loaded from Wikimedia Commons.")
with st.sidebar:
    st.markdown("## 🧭 TravelMate")
    st.caption("Karnataka travel • Retrieval-Augmented Generation")
    st.markdown("---")
    st.markdown("**RAG pipeline**")
    st.caption("Question → semantic embedding → ChromaDB retrieval → Gemini grounded generation → cited passages")
    if st.button("Warm up / rebuild knowledge index",use_container_width=True):
        try:
            with st.spinner("Embedding travel guide entries…"): count=refresh_index()
            st.success(f"{count} passages indexed.")
        except Exception as e: st.error(str(e))
    st.caption("The first semantic search builds the index automatically.")
tabs=st.tabs(["💬 AI Assistant","🗺️ Explore","🔎 Knowledge Search","📢 Travel Updates","❓ Ask a Question","🔐 Admin"])
with tabs[0]: chat_tab()
with tabs[1]: explore_tab()
with tabs[2]: search_tab()
with tabs[3]: updates_tab()
with tabs[4]: ask_tab()
with tabs[5]: admin_tab()
st.markdown("---")
st.caption("ಜೈ ಕರ್ನಾಟಕ! · TravelMate provides general planning information, not live booking or emergency services. Always confirm time-sensitive details with official providers.")
