from text_preprocessing import open_and_read_pdf,chunkify,split_chunk_to_individual_registers,filter_chunks_lower_bound#,get_ideal_chunk_size,
from embeddings import create_embeddings, semantic_search
from utils import preprocess_loaded_embeddings,preprocess_created_embeddings
import os.path
import pandas as pd
import torch
#change these values accordingly to your use case
page_start = 41 
input_path = "source_material/human-nutrition-text.pdf"
initial_chunk_size = 10
max_token_length = 384 
min_token_length = 30
embedder_model = 'all-mpnet-base-v2' 
token_threshold = 85
batch_size = 1
embeddings_path = "embeddings/text_chunks_embeddings.csv"
device = 'cuda' if torch.cuda.is_available() else "cpu"

## Check embedding pipeline upstream for errors
# Read and Preprocess Text
pages_and_texts = open_and_read_pdf(pdf_path=input_path,page_start = page_start)
chunkify(pages_and_texts, initial_chunk_size)
pages_and_chunks = split_chunk_to_individual_registers(pages_and_texts)
pages_and_chunks = filter_chunks_lower_bound(pages_and_chunks,min_token_length)

#Create text embedding if not yet exist.
if os.path.isfile(embeddings_path):
    print('captured here')
    pages_and_chunks, embeddings = preprocess_loaded_embeddings(embeddings_path)
else:
    embeddings_df = create_embeddings(chunk_dict = pages_and_chunks, model=embedder_model, save_path=embeddings_path, batch_size=batch_size, save_embeddings_to_file=True)
    pages_and_chunks, embeddings = preprocess_created_embeddings(embeddings_path)
#Respond to user query
query = 'which food is best for protein intake?'
print(semantic_search(query=query, top_k=5, source_embedding=embeddings, model=embedder_model, device=device))