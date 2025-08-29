import os.path
import chromadb
import requests
import sys
from sentence_transformers import SentenceTransformer

try: import tomllib
except ModuleNotFoundError: import tomli as tomllib

from process_text import create_database

if "__main__" == __name__:
    with open("pyproject.toml", mode="rb") as fp:
        args = tomllib.load(fp)

    print("Wczytywanie danych...")
    if not os.path.exists('./data/chroma.sqlite3'):
        print("Nie znaleziono bazy danych, tworzenie osadzeń plików txt w ./data/ ...")
        client = create_database("./data/")
    else:
        client = chromadb.PersistentClient(path="./data")
    collection = client.get_collection(name="bielik_rag")

    print("Wczytywanie tokenizera...\n")
    model = SentenceTransformer('ipipan/silver-retriever-base-v1')

    print("Witaj w systemie RAG, jak mogę Ci pomóc?")
    while True:
        query = input("> ")
        question_embeddings = model.encode(query, convert_to_numpy=True)
        matching_entries = collection.query(query_embeddings=question_embeddings, n_results=5)

        try:
            response = requests.post(
                url=args["ollama"]["api"],
                headers={"Content-Type": "application/json"},
                json={
                    "model": "SpeakLeash/bielik-11b-v2.2-instruct:Q4_K_M",
                    "stream": False,
                    "messages": [
                        {"role": "system",
                         "content": args['ollama']['system_prompt'] + ''.join(matching_entries['documents'][0])
                         },
                        {"role": "user", "content": query}
                    ],
                }
            )

        except requests.exceptions.ConnectionError:
            print("Nie udało połączyć się z modelem, zamykam program...")
            sys.exit(-1)

        if 200 == response.status_code:
            print(response.json()['message']['content'])
        else:
            print(f"Nie udało się uzyskać odpowiedzi, status: {response.status_code}, content: {response.text}")
            sys.exit(-1)