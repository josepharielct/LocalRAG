from sentence_transformers import util, SentenceTransformer
import torch
from tqdm import tqdm
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else "cpu"


def create_embeddings(chunk_dict:list[dict], model:str, save_path:str, batch_size:int = 1, save_embeddings_to_file:bool = True):
    embedding_model = SentenceTransformer(model_name_or_path = model,
                                            device = device)
    embedding_model.to(device)
    for item in tqdm(chunk_dict):
        item['embedding'] = embedding_model.encode(item['sentence_chunk'],
                                                   batch_size=batch_size,
                                                   convert_to_tensor=True)
    text_chunk_embeddings = pd.DataFrame(chunk_dict)
    if save_embeddings_to_file == True:
        embeddings_df_save_path = save_path
        text_chunk_embeddings.to_csv(embeddings_df_save_path, index=False)
    return text_chunk_embeddings

def semantic_search(query:str, top_k:int, source_embedding, model:str,device:str = device):
    embedding_model = SentenceTransformer(model_name_or_path = 'all-mpnet-base-v2',
                                        device=device)
    query = query
    print(f"Query: {query}")

    query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)
    print(query_embedding.shape)
    print(source_embedding.shape)
    dot_scores = util.dot_score(a=query_embedding, b=source_embedding)[0]
    top_results_dot_product = torch.topk(dot_scores,k=top_k)
    return top_results_dot_product