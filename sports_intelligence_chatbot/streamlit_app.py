import os
import io
import streamlit as st
from typing import List
from src.sports_chatbot import SportsChatbot
from src.decision_engine import DecisionEngine

st.set_page_config(page_title="Sports Intelligence Chatbot", page_icon="⚽", layout="wide")

DEFAULT_KB_DIR = os.path.join(os.path.dirname(__file__), "data", "sports_knowledge_base")

if "bot" not in st.session_state:
	st.session_state.bot = None
if "kb_dir" not in st.session_state:
	st.session_state.kb_dir = DEFAULT_KB_DIR
if "history" not in st.session_state:
	st.session_state.history = []  # list of dicts: {role: "user"|"assistant", content: str}

st.title("Sports Intelligence Chatbot")

with st.sidebar:
	st.subheader("Knowledge Base")
	kb_dir = st.text_input("Knowledge base directory", value=st.session_state.kb_dir)
	uploaded = st.file_uploader("Add documents (PDF/TXT/MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)
	if uploaded:
		os.makedirs(kb_dir, exist_ok=True)
		for uf in uploaded:
			path = os.path.join(kb_dir, uf.name)
			with open(path, "wb") as f:
				f.write(uf.getbuffer())
		st.success(f"Saved {len(uploaded)} file(s) to {kb_dir}")
	col1, col2 = st.columns(2)
	with col1:
		if st.button("Rebuild index"):
			with st.spinner("Building index..."):
				st.session_state.bot = SportsChatbot(kb_dir)
				st.session_state.kb_dir = kb_dir
				st.session_state.history = []
				st.success("Index ready")
	with col2:
		strategy_override = st.selectbox("Response strategy override (optional)", ["auto", "factual", "comparative", "analytical", "creative"]) 

# Render chat history
for msg in st.session_state.history:
	if msg["role"] == "user":
		st.chat_message("user").write(msg["content"])
	else:
		st.chat_message("assistant").write(msg["content"])

if st.session_state.bot is None:
	st.info("Build the index from the sidebar to get started.")
	st.stop()

prompt = st.chat_input("Ask a sports question...")
if prompt:
	st.session_state.history.append({"role": "user", "content": prompt})
	with st.spinner("Thinking..."):
		resp = st.session_state.bot.ask(prompt)
		answer = resp.get("answer")
		meta = f"(strategy={resp.get('strategy')}, conf={resp.get('confidence'):.2f})"
		full_answer = f"{answer}\n\n{meta}"
		st.session_state.history.append({"role": "assistant", "content": full_answer})
		st.rerun() 