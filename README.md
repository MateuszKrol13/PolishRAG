# RAG for Polish Users
This project aims to set up a local RAG console application for querying documentation (in txt files) in Polish language.
Because of those constraints, Bielik SLM was chosen as target model, since percentage of Polish language data used in GPT
and other english-based models is *far below one percent* [citation needed]. For deriving embeddings - again, 
for Polish language data - Silver Retriever Embedder is used.

Communication between user and model happens over http requests, so with minor modifications the project can be extended
to web agent.

## Stack
* Python >= 3.10
* Ollama for models
* Bielik SLM: SpeakLeash/bielik-11b-v2.2-instruct:Q4_K_M
* Embedder: ipipan/silver-retriever-base-v1
* ChromaDb with cosine similarity for embeddings storage

## How to run?
You need to [set up ollama](https://ollama.com/download/windows). After you finish setting up ollama open command prompt
, download the model and run ollama server:

    ollama pull SpeakLeash/bielik-11b-v2.2-instruct:Q4_K_M
    ollama serve

Activate python enviroment and run:

    pip install -r requirements.txt

After virtual enviroment finishes downloading necessary packages you can run setup script *process_text.py* to speed up 
database setup. Again, not necessary but first run will be slowed down.

Finally run `py rag_pipeline.py` and follow the instructions on command prompt.
## Notes
* Adjust values in *pyproject.toml* to change model and host address / port. 
* Ollama by default uses port:11434, which is not the one used in this project

## Findings
* It seems that system prompt "*Odpowiedz zwięźle na pytanie na podstawie tekstu:*" works well. Without constraints on 
the answer, model reffered to outside resources
