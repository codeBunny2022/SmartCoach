import os
from typing import List, Dict
from .document_processor import DocumentProcessor
from .vector_store import EmbeddingModel, FAISSVectorStore
from .rag_strategies import RAGContext
from .decision_engine import DecisionEngine
from .utils import simple_logger

try:
	from transformers import AutoModelForCausalLM, AutoTokenizer
	export_torch = True
except Exception:
	export_torch = False
	AutoModelForCausalLM = None
	AutoTokenizer = None


class SportsChatbot:
	def __init__(self, knowledge_dir: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", llm_name: str = "sshleifer/tiny-gpt2"):
		self.processor = DocumentProcessor()
		self.embedder = EmbeddingModel(embedding_model)
		self.docs: List[Dict] = []
		self.store: FAISSVectorStore = None  # type: ignore
		self.ctx: RAGContext = None  # type: ignore
		self.engine: DecisionEngine = None  # type: ignore
		self.llm_name = llm_name
		self.llm = None
		self.tokenizer = None
		self._build(knowledge_dir)

	def _build(self, knowledge_dir: str) -> None:
		simple_logger("Loading and chunking documents...")
		self.docs = self.processor.load_and_chunk(knowledge_dir)
		texts = [d["text"] for d in self.docs]
		simple_logger("Computing embeddings...")
		emb = self.embedder.embed(texts) if texts else self.embedder.embed([""])
		self.store = FAISSVectorStore(embedding_dim=emb.shape[1])
		if texts:
			self.store.add(emb, self.docs)
		self.ctx = RAGContext(self.embedder, self.store)
		self.engine = DecisionEngine(self.ctx)
		if export_torch:
			try:
				simple_logger(f"Loading local LLM: {self.llm_name}")
				self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
				self.llm = AutoModelForCausalLM.from_pretrained(self.llm_name)
				if getattr(self.llm.config, "pad_token_id", None) is None:
					self.llm.config.pad_token_id = getattr(self.llm.config, "eos_token_id", None) or self.tokenizer.eos_token_id
			except Exception as e:
				simple_logger(f"Warning: could not load LLM {self.llm_name}: {e}")

	def _extractive_answer(self, query: str, contexts: List[str]) -> str:
		# Simple extractive synthesis: pick the most relevant sentence from top contexts
		sentences: List[str] = []
		for ctx in contexts[:3]:
			for s in ctx.split(". "):
				if len(s.strip()) > 20:
					sentences.append(s.strip())
		if not sentences:
			return contexts[0][:400]
		# Rank sentences by embedding cosine with query
		q_emb = self.embedder.embed([query])
		s_emb = self.embedder.embed(sentences)
		scores = (s_emb @ q_emb.T).reshape(-1)
		best_idx = int(scores.argmax())
		return sentences[best_idx][:400]

	def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
		if self.llm is None or self.tokenizer is None:
			return prompt.split("Context:\n")[-1][-600:]
		model_max = getattr(self.llm.config, "n_positions", None)
		if model_max is None or model_max > 2048:
			model_max = min(self.tokenizer.model_max_length, 2048)
		max_input_tokens = max(256, model_max - max_new_tokens - 8)
		inputs = self.tokenizer(
			prompt,
			return_tensors="pt",
			truncation=True,
			max_length=max_input_tokens,
		)
		outputs = self.llm.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			pad_token_id=self.llm.config.pad_token_id,
		)
		text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		return text[-800:]

	def ask(self, query: str) -> Dict:
		decision = self.engine.route(query)
		if decision.get("strategy") == "reject-non-sport":
			return decision
		contexts = decision.get("contexts", [])
		citations = decision.get("citations", [])
		if not contexts:
			return {
				"answer": "I don't have enough information in the knowledge base to answer that confidently.",
				"confidence": decision.get("confidence", 0.3),
				"citations": [],
				"strategy": decision.get("strategy"),
			}
		top_n = min(3, len(citations))
		snippets = []
		for i in range(top_n):
			c = citations[i]
			snippet = contexts[i][:400]
			snippets.append(f"Source[{i+1}]: {c['source']}#chunk{c['chunk_index']}\n{snippet}")
		context_text = "\n\n".join(snippets)
		prompt = (
			"You are a concise, accurate sports expert. Answer the user using only the provided context. "
			"Cite sources as [S<number>]. If unsure, say you don't know.\n\n"
			f"Context:\n{context_text}\n\n"
			f"Question: {query}\nAnswer: "
		)
		# Use tiny LLM only when contexts are short; otherwise extractive answer
		if sum(len(s) for s in contexts[:top_n]) > 900:
			answer = self._extractive_answer(query, contexts)
		else:
			answer = self._generate(prompt)
		return {
			"answer": answer,
			"confidence": decision.get("confidence", 0.5),
			"citations": citations[:top_n],
			"strategy": decision.get("strategy"),
		} 