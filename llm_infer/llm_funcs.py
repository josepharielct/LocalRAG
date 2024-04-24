from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import torch 
from embeddings import retrieve_resources

def get_model_and_config_for_device():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")
    if gpu_memory_gb < 5.1:
        print('not enough gpu memory to run gemma locally')
    elif gpu_memory_gb < 8.1:
        use_quantization_config = True 
        model_id = "google/gemma-2b-it"
    elif gpu_memory_gb < 19.0:
        use_quantization_config = False 
        model_id = "google/gemma-2b-it"
    elif gpu_memory_gb > 19.0:
        use_quantization_config = False 
        model_id = "google/gemma-7b-it"

    print(f"use_quantization_config: {use_quantization_config}")
    print(f"model_id: {model_id}")
    return use_quantization_config, model_id

def get_quant_config():
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)
    return quantization_config

def get_attn_implementation():
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0]>=8):
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'sdpa'
    return attn_implementation


def generate_llm():
    use_quantization_config, model = get_model_and_config_for_device()
    attn_implementation = get_attn_implementation()
    quantization_config = get_quant_config()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model)
    llm_model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model,
                                                torch_dtype=torch.float16,
                                                quantization_config=quantization_config if use_quantization_config else None,
                                                low_cpu_mem_usage = False,
                                                attn_implementation=attn_implementation)
    if not use_quantization_config:
        llm_model.to('cuda')

    return tokenizer,llm_model


def format_prompt(query: str, context_items: list[dict], tokenizer)-> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
                    Give yourself room to think by extracting relevant passages from the context before answering the query.
                    Don't return the thinking, only return the answer.
                    Make sure your answers are as explanatory as possible.
                    Use the following examples as reference for the ideal answer style.
                    \nExample 1:
                    Query: What are the fat-soluble vitamins?
                    Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
                    \nExample 2:
                    Query: What are the causes of type 2 diabetes?
                    Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
                    \nExample 3:
                    Query: What is the importance of hydration for physical performance?
                    Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
                    \nNow use the following context items to answer the user query:
                    {context}
                    \nRelevant passages: <extract relevant passages from the context here>
                    User query: {query}
                    Answer:""" 
    base_prompt = base_prompt.format(context=context,
                                     query=query)
    dialogue_template = [
        {"role": "user",
         "content": base_prompt}
    ]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt

def ask(query: str,
        pages_and_chunks,
        embeddings,
        embedding_model,
        tokenizer,
        llm_model,
        temperature: float=0.7,
        max_new_tokens:int=256,
        format_answer_text=True,
        return_answer_only=True,
        ):
    scores, indices = retrieve_resources(query=query,
                                         top_k=5,
                                         source_embedding=embeddings,
                                         model=embedding_model,
                                         )

    context_items = [pages_and_chunks[i] for i in indices] 

    for i, item in enumerate(context_items): 
        item["score"] = scores[i].cpu()

    prompt = format_prompt(query=query,
                              context_items=context_items,
                              tokenizer=tokenizer)

    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0])
    if format_answer_text:
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "")
    if return_answer_only:
        return output_text

    return output_text, context_items