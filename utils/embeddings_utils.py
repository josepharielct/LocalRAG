import pandas as pd 
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else "cpu"
def preprocess_loaded_embeddings(file_path:str):
    text_chunks_embeddings = pd.read_csv(file_path)
    text_chunks_embeddings['embedding'] = text_chunks_embeddings['embedding'].apply(lambda x: np.fromstring(x.strip("[]"),sep= ' '))
    embeddings = torch.tensor(np.stack(text_chunks_embeddings['embedding'].tolist(),axis=0),dtype=torch.float32).to(device)
    pages_and_chunks = text_chunks_embeddings.to_dict(orient='records')
    print(embeddings.shape)
    return pages_and_chunks, embeddings
def preprocess_created_embeddings(embeddings_df):
   embeddings = torch.tensor(np.stack(embeddings_df['embedding'].tolist(),axis=0),dtype=torch.float32).to(device)
   pages_and_chunks = embeddings_df.to_dict(orient='records')
   return pages_and_chunks, embeddings

