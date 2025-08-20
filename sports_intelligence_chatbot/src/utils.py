import os
import re
import json
import time
from typing import Iterable, List, Tuple, Dict, Optional


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_logger(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {message}")


def split_text_with_overlap(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    tokens = re.split(r"(\n\n+|\.)", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for t in tokens:
        if t is None:
            continue
        add_len = len(t)
        if current_len + add_len >= chunk_size and current:
            chunks.append("".join(current).strip())
            if overlap > 0 and chunks[-1]:
                tail = chunks[-1][-overlap:]
                current = [tail, t]
                current_len = len(tail) + add_len
            else:
                current = [t]
                current_len = add_len
        else:
            current.append(t)
            current_len += add_len
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def time_it(label: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000.0
            simple_logger(f"{label} took {duration:.1f} ms")
            return result
        return wrapper
    return decorator


class LRUCache:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.cache: Dict[str, Tuple[float, object]] = {}
        self.order: List[str] = []

    def get(self, key: str) -> Optional[object]:
        if key in self.cache:
            self.order.remove(key)
            self.order.insert(0, key)
            return self.cache[key][1]
        return None

    def set(self, key: str, value: object) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.order) >= self.capacity:
            evict = self.order.pop()
            self.cache.pop(evict, None)
        self.order.insert(0, key)
        self.cache[key] = (time.time(), value) 