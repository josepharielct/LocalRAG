from text_preprocessing import open_and_read_pdf,chunkify,split_chunk_to_individual_registers,filter_chunks_lower_bound
from embeddings import create_embeddings,preprocess_loaded_embeddings
import os.path
import torch
from sentence_transformers import SentenceTransformer
from llm_infer import generate_llm, ask
import gradio as gr

#change these values accordingly to your use case
page_start = 41 
input_path = "source_material/human-nutrition-text.pdf"
chunk_size = 10
max_token_length = 384 
min_token_length = 30
embedder_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                     device="cpu")
token_threshold = 85
batch_size = 1
embeddings_path = "embeddings/text_chunks_embeddings.csv"
device = 'cuda' if torch.cuda.is_available() else "cpu"

#Create text embedding if not yet exist.
if os.path.isfile(embeddings_path):
    pages_and_chunks, embeddings = preprocess_loaded_embeddings(embeddings_path)
else:
        # Read and Preprocess Text
        pages_and_texts = open_and_read_pdf(pdf_path=input_path,page_start = page_start)
        chunkify(pages_and_texts, chunk_size)
        pages_and_chunks = split_chunk_to_individual_registers(pages_and_texts)
        pages_and_chunks = filter_chunks_lower_bound(pages_and_chunks,min_token_length)
        #Create embeddings
        embeddings_df = create_embeddings(chunk_dict = pages_and_chunks, model=embedder_model, save_path=embeddings_path, batch_size=batch_size)
        pages_and_chunks, embeddings = preprocess_loaded_embeddings(embeddings_path)

#Respond to user query
tokenizer, llm_model = generate_llm()

def get_response(message, history):
    return ask(query=message, pages_and_chunks=pages_and_chunks, embeddings=embeddings,embedding_model=embedder_model,tokenizer=tokenizer,llm_model=llm_model)
gr.ChatInterface(get_response).launch(share=True)