from typing import Literal
import re

QueryType = Literal["factual", "comparative", "analytical", "creative", "nonsport"]

try:
	from transformers import pipeline  # type: ignore
	hf_available = True
except Exception:
	hf_available = False

# Broader sport cues including cricket terms
SPORT_PATTERNS = [
	r"\bfootball\b", r"\bsoccer\b", r"\bcricket\b", r"\bbasketball\b", r"\btennis\b",
	r"\bipl\b", r"\bfifa\b", r"\bnba\b", r"premier league",
	r"off\s*-?\s*side", r"no\s*-?\s*ball", r"free\s*hit", r"\bwide\b", r"\blbw\b",
	r"\bwicket\b", r"\bbowler\b", r"\bbatting\b", r"\bserve\b", r"\bgoal\b", r"red\s*card"
]

FACTUAL_PREFIXES = (
	"what is", "what's", "define", "explain", "when is", "when was", "who is", "who was",
	"what happens", "how is", "how does", "tell me about",
)
COMPARATIVE_CUES = ("compare", "vs", "versus", "head-to-head", "better than")
ANALYTICAL_CUES = ("why", "strategy", "tactical", "analyze", "analysis", "reason")
CREATIVE_CUES = ("suggest", "recommend", "what if", "synthesize", "insight", "novel")


class ZeroShotQueryClassifier:
	def __init__(self):
		self.labels = ["factual", "comparative", "analytical", "creative", "nonsport"]
		self.zs = None
		if hf_available:
			try:
				self.zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
			except Exception:
				self.zs = None

	def _is_probably_sport(self, text: str) -> bool:
		lower = text.lower()
		for pat in SPORT_PATTERNS:
			if re.search(pat, lower):
				return True
		return False

	def classify(self, text: str) -> QueryType:
		q = text.strip()
		lq = q.lower()
		# High-precision overrides first
		if lq.startswith(FACTUAL_PREFIXES):
			if self._is_probably_sport(lq):
				return "factual"
		if any(c in lq for c in COMPARATIVE_CUES) and self._is_probably_sport(lq):
			return "comparative"
		if any(c in lq for c in ANALYTICAL_CUES) and self._is_probably_sport(lq):
			return "analytical"
		if any(c in lq for c in CREATIVE_CUES) and self._is_probably_sport(lq):
			return "creative"

		if self.zs is None:
			if not self._is_probably_sport(q):
				return "nonsport"
			return "factual"
		# Zero-shot primary
		res = self.zs(q, candidate_labels=self.labels, multi_label=False)
		label = res["labels"][0]
		return label  # type: ignore


def classify_query(text: str) -> QueryType:
	return ZeroShotQueryClassifier().classify(text) 