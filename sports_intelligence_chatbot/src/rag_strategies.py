from typing import List, Dict, Tuple
import numpy as np
from .utils import simple_logger

try:
	from transformers import pipeline  # type: ignore
	hf_available = True
except Exception:
	hf_available = False


class RAGContext:
	def __init__(self, embedder, store):
		self.embedder = embedder
		self.store = store


def _format_citations(hits: List[Tuple[int, float]], metadatas: List[Dict]) -> List[Dict]:
	citations = []
	for idx, score in hits:
		if idx < 0:
			continue
		meta = metadatas[idx]
		citations.append({
			"source": meta.get("source"),
			"chunk_index": meta.get("chunk_index"),
			"score": round(float(score), 4),
		})
	return citations


def simple_rag(ctx: RAGContext, query: str, k: int = 5, min_score: float = 0.2) -> Dict:
	emb = ctx.embedder.embed([query])
	results = ctx.store.search(emb, k=k)[0]
	filtered = [(i, s) for i, s in results if s >= min_score]
	metas = ctx.store.metadatas
	contexts = [metas[i]["text"] for i, _ in filtered]
	citations = _format_citations(filtered, metas)
	return {"contexts": contexts, "citations": citations}


def hierarchical_rag(ctx: RAGContext, query: str, k: int = 10, summary_top_k: int = 4) -> Dict:
	first = simple_rag(ctx, query, k=k)
	texts = first["contexts"]
	if not texts:
		return first
	summaries = [t[:300] for t in texts]
	summary_emb = ctx.embedder.embed(summaries)
	query_emb = ctx.embedder.embed([query])
	scores = (summary_emb @ query_emb.T).reshape(-1)
	order = np.argsort(-scores)[:summary_top_k]
	selected_contexts = [texts[i] for i in order]
	selected_hits = [first["citations"][i] for i in order]
	return {"contexts": selected_contexts, "citations": selected_hits}


def multihop_rag(ctx: RAGContext, query: str, hops: int = 2, k: int = 5) -> Dict:
	acc_contexts: List[str] = []
	acc_citations: List[Dict] = []
	current_query = query
	for _ in range(hops):
		res = simple_rag(ctx, current_query, k=k)
		if not res["contexts"]:
			break
		acc_contexts.extend(res["contexts"])
		acc_citations.extend(res["citations"])
		current_query = current_query + " " + res["contexts"][0][:200]
	return {"contexts": acc_contexts, "citations": acc_citations}


def adaptive_threshold_rag(ctx: RAGContext, query: str, base_k: int = 5) -> Dict:
	for threshold in [0.35, 0.3, 0.25, 0.2, 0.1]:
		res = simple_rag(ctx, query, k=base_k, min_score=threshold)
		if res["contexts"]:
			res["threshold"] = threshold
			return res
	return simple_rag(ctx, query, k=base_k, min_score=0.0)


def multiquery_rag(ctx: RAGContext, query: str, k_per_variant: int = 3) -> Dict:
	variants: List[str]
	if hf_available:
		try:
			qp = pipeline("text2text-generation", model="google/flan-t5-base")
			prompt = (
				"Generate 4 diverse rephrasings for information retrieval about the following sports question.\n"
				f"Question: {query}"
			)
			out = qp(prompt, max_new_tokens=64, num_return_sequences=1)[0]["generated_text"]
			variants = [query] + [v.strip("- •\n ") for v in out.split("\n") if len(v.strip()) > 0][:4]
		except Exception:
			variants = [query]
	else:
		variants = [query]
	all_hits: List[Tuple[int, float]] = []
	for v in variants:
		res = simple_rag(ctx, v, k=k_per_variant)
		metas = ctx.store.metadatas
		for i, s in [(i, s) for i, s in ctx.store.search(ctx.embedder.embed([v]), k=k_per_variant)[0]]:
			all_hits.append((i, s))
	# Deduplicate by index, keep max score
	score_map: Dict[int, float] = {}
	for i, s in all_hits:
		score_map[i] = max(s, score_map.get(i, -1e9))
	sorted_hits = sorted(score_map.items(), key=lambda x: -x[1])[:5]
	metas = ctx.store.metadatas
	contexts = [metas[i]["text"] for i, _ in sorted_hits]
	citations = _format_citations(sorted_hits, metas)
	return {"contexts": contexts, "citations": citations}


# HyDE removed: keep a no-op wrapper for compatibility

def hyde_rag(ctx: RAGContext, query: str, k: int = 5) -> Dict:
	return simple_rag(ctx, query, k=k) 