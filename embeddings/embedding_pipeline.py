from sentence_transformers import util, SentenceTransformer
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import textwrap
device = 'cuda' if torch.cuda.is_available() else "cpu"


def create_embeddings(chunk_dict:list[dict], model:str, save_path:str, batch_size:int = 1):
    embedding_model = model
    embedding_model.to(device)
    for item in tqdm(chunk_dict):
        item['embedding'] = embedding_model.encode(item['sentence_chunk'],
                                                   batch_size=batch_size,
                                                   convert_to_tensor=True)
    text_chunk_embeddings = pd.DataFrame(chunk_dict)
    embeddings_df_save_path = save_path
    text_chunk_embeddings.to_csv(embeddings_df_save_path, index=False)
    return text_chunk_embeddings

def semantic_search(query:str, top_k:int, source_embedding, model:str,page_chunk:list[dict], device:str = device):
    embedding_model = model
    query = query
    print(f"Query: {query}")

    query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)
    dot_scores = util.dot_score(a=query_embedding, b=source_embedding)[0]
    top_results_dot_product = torch.topk(dot_scores,k=top_k)
    results = []
    for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
        result = {
            'score': score,
            'idx': idx,
            'text': page_chunk[idx]['sentence_chunk'],
            'page_number': page_chunk[idx]['page_number']
        }
        results.append(result)
    return results

def retrieve_resources(query:str, top_k:int, source_embedding, model:str, device:str = device):
    embedding_model = model
    query = query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)
    dot_scores = util.dot_score(a=query_embedding, b=source_embedding)[0]
    scores,indexes = torch.topk(dot_scores,k=top_k)
    return scores,indexes


def preprocess_loaded_embeddings(file_path:str):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    text_chunks_embeddings = pd.read_csv(file_path)
    text_chunks_embeddings['embedding'] = text_chunks_embeddings['embedding'].apply(lambda x: x.replace('tensor(', '').replace(')', ''))
    text_chunks_embeddings['embedding'] = text_chunks_embeddings['embedding'].apply(lambda x: x.replace('\n', ''))
    text_chunks_embeddings['embedding'] = text_chunks_embeddings['embedding'].apply(lambda x: np.array(eval(x)))
    embeddings = torch.tensor(np.stack(text_chunks_embeddings['embedding'].tolist(),axis=0),dtype=torch.float32).to(device)
    print(embeddings.shape)
    pages_and_chunks = text_chunks_embeddings.to_dict(orient='records')
    return pages_and_chunks, embeddings

def print_wrapped(text,wrap_length=80):
  wrapped_text=textwrap.fill(text,wrap_length)
  print(wrapped_text)