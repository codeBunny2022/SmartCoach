import os
import sys
from src.sports_chatbot import SportsChatbot


def main():
	base = os.path.dirname(os.path.abspath(__file__))
	knowledge_dir = os.path.join(base, "../data/sports_knowledge_base")
	print("Loading Sports Intelligence Chatbot...")
	bot = SportsChatbot(knowledge_dir)
	print("Ready. Ask sports questions. Type 'exit' to quit.\n")
	while True:
		try:
			q = input("You: ").strip()
			if q.lower() in {"exit", "quit"}:
				break
			resp = bot.ask(q)
			print(f"\nAnswer (conf={resp.get('confidence'):.2f}, strategy={resp.get('strategy')}):\n{resp.get('answer')}\n")
			if resp.get("citations"):
				print("Sources:")
				for i, c in enumerate(resp["citations"], 1):
					print(f"  [S{i}] {c['source']}#chunk{c['chunk_index']} (score={c['score']})")
			print()
		except KeyboardInterrupt:
			break


if __name__ == "__main__":
	main() 