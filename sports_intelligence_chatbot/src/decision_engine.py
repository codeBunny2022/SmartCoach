from typing import Dict
from .query_classifier import classify_query
from .rag_strategies import RAGContext, simple_rag, hierarchical_rag, multihop_rag, adaptive_threshold_rag, multiquery_rag, hyde_rag
from .utils import simple_logger


class DecisionEngine:
	def __init__(self, ctx: RAGContext):
		self.ctx = ctx

	def _estimate_confidence(self, num_contexts: int) -> float:
		if num_contexts == 0:
			return 0.0
		if num_contexts < 2:
			return 0.6
		if num_contexts < 5:
			return 0.75
		return 0.85

	def route(self, query: str) -> Dict:
		qtype = classify_query(query)
		if qtype == "nonsport":
			return {
				"answer": "I am focused on sports-related questions. Could you rephrase your query within a sports context?",
				"confidence": 0.2,
				"citations": [],
				"strategy": "reject-non-sport",
			}
		if qtype == "factual":
			# Prefer adaptive simple retrieval for precision over HyDE with tiny LLM
			res = adaptive_threshold_rag(self.ctx, query)
		elif qtype == "comparative":
			mq = multiquery_rag(self.ctx, query)
			res = mq if len(mq.get("contexts", [])) >= 2 else hierarchical_rag(self.ctx, query)
		elif qtype == "analytical":
			res = multihop_rag(self.ctx, query, hops=2)
		else:  # creative
			res = multiquery_rag(self.ctx, query)

		confidence = self._estimate_confidence(len(res.get("contexts", [])))
		return {
			"contexts": res.get("contexts", []),
			"citations": res.get("citations", []),
			"confidence": confidence,
			"strategy": qtype,
		} 