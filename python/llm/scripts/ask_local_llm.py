import sys
from python.llm.ollama_client import OllamaClient

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask_local_llm.py 'your question'")
        return

    prompt = sys.argv[1]

    client = OllamaClient()
    response = client.ask(prompt)

    print("\n=== RESPONSE ===\n")
    print(response)

if __name__ == "__main__":
    main()