from typing import List, Dict, Tuple
import os
import numpy as np
from .utils import simple_logger

try:
	import faiss  # type: ignore
	FAISS_AVAILABLE = True
except Exception:
	faiss = None  # type: ignore
	FAISS_AVAILABLE = False

try:
	from sentence_transformers import SentenceTransformer
except Exception:
	SentenceTransformer = None

try:
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.preprocessing import normalize as sk_normalize
except Exception:
	TfidfVectorizer = None
	sk_normalize = None


class EmbeddingModel:
	def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
		self._use_st = SentenceTransformer is not None
		self.model_name = model_name
		self._is_fitted = False
		if self._use_st:
			self.model = SentenceTransformer(model_name)
			simple_logger(f"Loaded embedding model: {model_name}")
		else:
			if TfidfVectorizer is None or sk_normalize is None:
				raise RuntimeError("Neither sentence-transformers nor scikit-learn is available for embeddings.")
			self.vectorizer = TfidfVectorizer(max_features=4096)
			simple_logger("Using TF-IDF fallback embeddings (scikit-learn)")

	def embed(self, texts: List[str]) -> np.ndarray:
		# Sentence-Transformers path
		if self._use_st:
			embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
			return np.array(embeddings, dtype=np.float32)
		# TF-IDF path: fit on first call (assumed corpus), transform thereafter (queries)
		if not self._is_fitted:
			X = self.vectorizer.fit_transform(texts)
			self._is_fitted = True
		else:
			X = self.vectorizer.transform(texts)
		X = sk_normalize(X, norm="l2", copy=False)
		return X.toarray().astype(np.float32)


class FAISSVectorStore:
	def __init__(self, embedding_dim: int):
		self.embedding_dim = embedding_dim
		self.metadatas: List[Dict] = []
		self._use_faiss = FAISS_AVAILABLE
		if self._use_faiss:
			self.index = faiss.IndexFlatIP(embedding_dim)
		else:
			self.index = None
			self._matrix = np.empty((0, embedding_dim), dtype=np.float32)
			simple_logger("FAISS not available, using numpy fallback index")

	def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
		if embeddings.shape[1] != self.embedding_dim:
			raise ValueError("Embedding dimension mismatch")
		if self._use_faiss:
			self.index.add(embeddings)
		else:
			self._matrix = np.vstack([self._matrix, embeddings])
		self.metadatas.extend(metadatas)

	def search(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Tuple[int, float]]]:
		if query_embeddings.ndim == 1:
			query_embeddings = query_embeddings.reshape(1, -1)
		if self._use_faiss:
			scores, indices = self.index.search(query_embeddings, k)
		else:
			scores = query_embeddings @ self._matrix.T
			if scores.size == 0:
				indices = np.zeros((query_embeddings.shape[0], 0), dtype=int)
			else:
				indices = np.argsort(-scores, axis=1)[:, :k]
				scores = np.take_along_axis(scores, indices, axis=1) if indices.shape[1] > 0 else scores
		results: List[List[Tuple[int, float]]] = []
		if indices.size == 0:
			return [[] for _ in range(query_embeddings.shape[0])]
		for row_idx in range(indices.shape[0]):
			row: List[Tuple[int, float]] = []
			for col_idx in range(indices.shape[1]):
				idx = int(indices[row_idx, col_idx])
				score = float(scores[row_idx, col_idx])
				row.append((idx, score))
			results.append(row)
		return results

	def get_metadata(self, index: int) -> Dict:
		return self.metadatas[index] 