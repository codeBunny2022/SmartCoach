import os
from typing import List, Dict
from .utils import simple_logger, split_text_with_overlap

try:
	from pypdf import PdfReader
except Exception:
	PdfReader = None


class DocumentProcessor:
	def __init__(self, chunk_size: int = 800, overlap: int = 120):
		self.chunk_size = chunk_size
		self.overlap = overlap

	def _read_pdf(self, path: str) -> str:
		if PdfReader is None:
			raise RuntimeError("pypdf not available. Please install pypdf")
		reader = PdfReader(path)
		texts = []
		for page in reader.pages:
			texts.append(page.extract_text() or "")
		return "\n".join(texts)

	def _read_txt(self, path: str) -> str:
		with open(path, "r", encoding="utf-8", errors="ignore") as f:
			return f.read()

	def _read_md(self, path: str) -> str:
		return self._read_txt(path)

	def load_and_chunk(self, base_dir: str) -> List[Dict]:
		items: List[Dict] = []
		for root, _, files in os.walk(base_dir):
			for fname in files:
				path = os.path.join(root, fname)
				ext = os.path.splitext(fname)[1].lower()
				try:
					if ext == ".pdf":
						text = self._read_pdf(path)
					elif ext in (".txt", ".md"):
						text = self._read_txt(path)
					else:
						continue
					chunks = split_text_with_overlap(text, self.chunk_size, self.overlap)
					for i, chunk in enumerate(chunks):
						items.append({
							"id": f"{fname}:{i}",
							"source": path,
							"chunk_index": i,
							"text": chunk,
						})
					simple_logger(f"Processed {fname} -> {len(chunks)} chunks")
				except Exception as e:
					simple_logger(f"Error processing {fname}: {e}")
		return items 