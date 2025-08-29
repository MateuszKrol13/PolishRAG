# RAG for Polish Users
This project aims to set up a local RAG console application for querying documentation (in txt files) in polish language.  Because of those constraints, Bielik SLM was chosen as target model, since percentage of polish language data used in GPT and other english-based models is *far below one percent* [citation needed]. For deriving embeddings - again, for polish language data - Silver Retriever Embedder is used.

Communication between user and model happens over http requests, so with minor modifications the project can be extended to web agent.

## Stack
* Python >= 3.10
* Ollama for models
* Bielik SLM: SpeakLeash/bielik-11b-v2.2-instruct
* Embedder: ipipan/silver-retriever-base-v1
* ChromaDb with cosine similarity for embeddings storage
## How to run?
You need to [set up ollama](https://ollama.com/download/windows). You may then download the model by running ollama; this is not necessary as ollama will download any model that is not downloaded, but keep in mind that first run of this project will be greatly slowed down.

Activate python enviroment and run:

    pip instal requirements.txt

After virtual enviroment finishes downloading necessary packages you can run setup script *process_text.py* to speed up database setup. Again, not necessary but first run will be slowed down.

Finally run `python3 rag_pipeline.py` and follow the instructions on command prompt.
## Notes
* Adjust values in *pyproject.toml* to change model and host address / port. 
* Ollama by default uses port:11434, which is not the one used in this project

## Findings
* It seems that system prompt "*Odpowiedz zwięźle na pytanie na podstawie tekstu:*" works well. Without constraints on the answer, model reffered to outside resources
