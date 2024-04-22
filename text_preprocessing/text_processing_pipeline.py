import fitz # (pymupdf, found this is better than pypdf for our use case, note: licence is AGPL-3.0, keep that in mind if you want to use any code commercially)
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm
from spacy.lang.en import English
import re
import numpy as np
import pandas as pd


def text_formatter(text: str):
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def split_to_sentences(info_dict:list[dict]):
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(info_dict):
        item['sentences'] = list(nlp(item['text']).sents)
        item['sentences'] = [str(sentence) for sentence in item['sentences']]
        item['page_sentence_count'] = len(item['sentences'])

def split_list(input_list:list,
               chunk_size: int):
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]

def chunkify(info_dict:list[dict],chunk_size:int):
    for item in tqdm(info_dict):
        item['sentence_chunks'] = split_list(input_list=item['sentences'],
                                            chunk_size = chunk_size)
        item['num_chunks'] = len(item['sentence_chunks'])

def split_chunk_to_individual_registers(info_dict:list[dict]):
    pages_and_chunks = []
    for item in tqdm(info_dict):
        for sentence_chunk in item['sentence_chunks']:
            chunk_dict = {}
            chunk_dict['page_number'] = item['page_number']

            #join sentence together to paragraph
            joined_sentence_chunk = "".join(sentence_chunk).replace(" ", " ").strip()
            joined_sentence_chunk = re.sub(r"\.([A-Z])",r'. \1', joined_sentence_chunk)
            chunk_dict['sentence_chunk'] = joined_sentence_chunk
            chunk_dict['chunk_char_count'] = len(joined_sentence_chunk) 
            chunk_dict['chunk_word_count'] = len([word for word in joined_sentence_chunk.split(' ')])
            chunk_dict['chunk_token_count'] = len(joined_sentence_chunk) / 4

            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks
        

def open_and_read_pdf(pdf_path: str,page_start:int):
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number - page_start,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    split_to_sentences(pages_and_texts)
    return pages_and_texts

def get_token_count_threshold(info_dict:list[dict],threshold:int):
    return None
#     token_count = []
#     for item in tqdm(info_dict):
#         for sentence_chunk in item['sentence_chunks']:
#             joined_sentence_chunk = "".join(sentence_chunk).replace(" ", " ").strip()
#             joined_sentence_chunk = re.sub(r"\.([A-Z])",r'. \1', joined_sentence_chunk)
#             token_count.append(len(joined_sentence_chunk) / 4)
#     return np.percentile(token_count,threshold)


def get_ideal_chunk_size(info_dict: dict, initial_chunk_size: int, token_size_limit: int, token_threshold: int):
    return None
#     copy_dict = info_dict.copy()
#     current_size = initial_chunk_size
#     chunkify(copy_dict, current_size)
#     token_size_in_threshold = get_token_count_threshold(copy_dict,token_threshold)

#     while token_size_in_threshold > token_size_limit:
#         current_size -= 1
#         copy_dict = info_dict.copy()
#         chunkify(copy_dict, current_size)
#         token_size_in_threshold = get_token_count_threshold(copy_dict, token_threshold)
#         print(f"current_size: {current_size}, current_max_token_size: {token_size_in_threshold}")

#     return current_size


def filter_chunks_lower_bound(chunk_dict:list[dict], min_token_length:int):
    df = pd.DataFrame(chunk_dict)
    filtered_dict = df[df['chunk_token_count'] > min_token_length].to_dict(orient='records')
    return filtered_dict


