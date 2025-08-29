import os
import unicodedata
import numpy as np
import chromadb

from sentence_transformers import SentenceTransformer

def split_text_to_chunks(string_content: str, window_len: int, window_overlap: int) -> list[str]:
    """
    Splits string into chunks of specified length, so that the embedder can process each chunk without data loss

    Required Parameters:
        string_content (str): string to be split
        window_len (int): length of chunk
        window_overlap (int): specifies number of characters that overlap between previous and next chunk

    Returns:
        list[str]: list of text chunks
    """
    chunks = []
    for i in range(0, len(string_content), window_len - window_overlap):
        chunks.append(string_content[i:i + window_len])

    return chunks

def strip_unicode_control_chars(string_content: str) -> str:
    """Removes Unicode control characters from passed string"""
    return "".join(char for char in string_content if unicodedata.category(char) != "Cc")  # strip control characters

def chunks_to_embeddings(embedder: SentenceTransformer, text_chunks: list[str]):
    """Embeds each chunk

    Notes:
        Used embedder has maximal chunk size of 512, change if necessary
    """
    return [embedder.encode(chunk, chunk_size=512) for chunk in text_chunks]

if "__main__" == __name__:
    text_chunks = []
    text_files = [file_ for file_ in os.scandir("./docs/") if file_.name.endswith(".txt")]
    for file_ in text_files:
        with open(file_.path, "r", encoding='utf-8') as f:
            # Provided text files contained control characters, which deeply polluted the context
            content = strip_unicode_control_chars(f.read())
            text_chunks += split_text_to_chunks(content, window_len=512, window_overlap=128)

    model = SentenceTransformer('ipipan/silver-retriever-base-v1')
    if any(len(chunk) > model.max_seq_length for chunk in text_chunks):
        raise Warning(
            f"Some text chunks are longer than maximum sequence lenght of {model.max_seq_length}! "
            f"Portion of data is getting lost..."
        )
    embeddings = model.encode(text_chunks, convert_to_numpy=True)

    client = chromadb.PersistentClient(path="./docs/")
    collection = client.get_or_create_collection(name="bielik_rag", metadata={"hnsw:space": "cosine"})
    collection.add(ids=[str(i) for i in range(len(text_chunks))], documents=text_chunks, embeddings=embeddings)







