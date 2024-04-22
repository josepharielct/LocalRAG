from sentence_transformers import SentenceTransformer
import torch
import tqdm
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else "cpu"


def create_embeddings(chunk_dict:list[dict], model:str, save_path:str, batch_size:int = 1, save_embeddings_to_file:bool = True):
    embedding_model = SentenceTransformer(model_name_or_path = model,
                                            device = device)
    embedding_model.to(device)
    print(chunk_dict[0])
    for item in chunk_dict:
        item['embedding'] = embedding_model.encode(item['sentence_chunk'],
                                                   batch_size=batch_size,
                                                   convert_to_tensor=True)
    if save_embeddings_to_file == True:
        text_chunk_embeddings = pd.DataFrame(chunk_dict)
        embeddings_df_save_path = save_path
        text_chunk_embeddings.to_csv(embeddings_df_save_path, index=False)
    return text_chunk_embeddings

def rag_search_answer(embeddings_df,device):
    return None