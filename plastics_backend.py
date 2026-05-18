from collections import defaultdict
import pandas as pd
import faiss
import torch
from flask import Flask, request, jsonify, abort
from threading import Semaphore
import openai  # for using GPT and getting embeddings
import requests
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from retrieval_encode import BGE, GTE , RepLLaMA
import os 

# Download the stopwords list if not already downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')

app = Flask(__name__)
openai.api_key = "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"


# Limit to 1 concurrent call to avoid GPU OOM
semaphore = Semaphore(1)
global initMode
initModel = False

def set_index(fn):
    index = faiss.read_index(fn)
    if isinstance(index, faiss.IndexPreTransform):
        inner_index = faiss.downcast_index(index.index)
        if isinstance(inner_index, faiss.IndexIVF):
            inner_index.nprobe = 512
        if hasattr(inner_index, 'quantizer'):
            quantizer = faiss.downcast_index(inner_index.quantizer)
            if isinstance(quantizer, faiss.IndexHNSW):
                quantizer.hnsw.efSearch = 512
    elif isinstance(index, faiss.IndexIVF):
        index.nprobe = 512
        if hasattr(index, 'quantizer'):
            quantizer = faiss.downcast_index(index.quantizer)
            if isinstance(quantizer, faiss.IndexHNSW):
                quantizer.hnsw.efSearch = 512
    return index


def initialize_app():
    global models, metas, indexs

    models = {"bge": BGE(f"cpu"), "gte": GTE(f"cpu")}
    path_data_files = "/data_hdd/damian/plastics_sustainability/corpus/plasticsClass_json/merged_text"
    path_data_index = "/data_hdd/damian/plastics_sustainability/corpus/plasticsClass_emb/merged_emb"
    
    para_corpus = load_dataset("json", data_files= os.path.join(path_data_files , "merged.jsonl"), cache_dir=os.path.join(path_data_files,".cache"))
    metas = {
        "sentence": para_corpus["train"],
    }
    indexs = {
        "sentence": {
            "bge": set_index(os.path.join (path_data_index , "corpus_embedding.faiss")),
        },
    }



def retrieve_step(query, n_answers, environment_name ,dense):
    # Define the URL endpoint
    
    url = {
        "test" : "http://infodeep2.ece.udel.edu:8080/search/",
        "production" : "http://infochain.ece.udel.edu:8080/search/"
    }
    
    print("Q: " + query)
    # JSON input data
    input_data = {
        "test": {
            "yql": "select * from plastics where userQuery()",
            "query": f"{query}",
            "type": "any",
            "offset": 0,
            "hits": n_answers,
            "presentation.summary": "full",
            "ranking.profile": "bm25Content",
            "trace": {"level": 0},
        },

        "production":{
            "yql": "select * from article where userQuery() and !(titleSection contains 'Abstract')  and isAbstract = false",
            "query": f"{query}",
            "offset": 0,
            "hits": n_answers,
            "presentation.summary": "full",
            "ranking.profile": "bm25",
            "trace": {"level": 0},    
        }
    }

    input_dense = {
            "yql": 'select * from article where ({targetHits:100000}nearestNeighbor(embedding,q))',
            "input.query(q)": f"embed(Represent this sentence for searching relevant passages: {query})",
            "offset": 0,
            "hits": n_answers,
            "presentation.summary":"short",
            "ranking.profile": "semantic",
            "trace":{
                "level" : 0
            }          
    }
    # Convert input data to JSON format
    if dense == True:
        json_input = json.dumps(input_dense)
    else:
        json_input = json.dumps( input_data[environment_name])
    print("Request")
    #print(json_input)
    # Set the headers
    headers = {"Content-Type": "application/json"}

    # Make the POST request
    response = requests.post(url[environment_name], data=json_input, headers=headers)
    print("StepA")
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        output_data = response.json()
        # print("Response:", output_data)
    else:
        #print("Error:", response.status_code)
        output_data = response.json()
        #print("Response:", output_data)
    print(output_data)
    return output_data


def relevance_step(query, chunk_of_text):
    global tokenizer, model

    # if initModel == False:

    #     initModel = True

    def generate(prompt_reformed):
        inputs = tokenizer(prompt_reformed, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, use_cache=True, max_length=4096)
        output_text = tokenizer.decode(outputs[0])
        text_output = output_text.split("[/INST]Results:\n")[-1]
        text_output = text_output.split("</s>")[0]
        return text_output

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    assistant_prompt = (
        "Given a user request and a search result, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n\n"
        "3 = key, this search result contains relevant, diverse, informative and correct answers to the user request; the user request can be fulfilled by relying only on this search result.\n"
        "2 = high relevance, this search result contains relevant, informative and correct answers to the user request; however, it does not span diverse perspectives, and including another perspective can help with a better answer to the user request.\n"
        "1 = minimal relevance, this search result contains relevant answers to the user request. However, it is impossible to answer the user request based solely on the search result. \n"
        "0 = not relevant, this search result does not contain any relevant answer to the user request.\n"
        "Assume that you are collecting all the relevant search results to write a final answer for the user request.\n"
    )
    user_prompt = (
        "User Request:\n"
        "A user typed the following request.\n"
        "{query}.\n"
        "Result:\n"
        "Consider the following search result:\n"
        "—BEGIN Search Result CONTENT—\n"
        "{snippet}\n"
        "—END Search Result CONTENT—\n"
        "Instructions:\n"
        "Split this problem into steps:\n"
        "Consider the underlying intent of the user request.\n"
        "Measure how well the search result matches a likely intent of the request (M)\n"
        "Measure how trustworthy the search result is (T).\n"
        "Consider the aspects above and the relative importance of each, and decide on a final score (O).\n"
        'Produce a JSON of scores without providing any reasoning. Example:{"M": 2, "T": 1, "O": 1}\n'
        "NO PROVIDE ANY Justification ONLY THE JSON\n"
    )

    labeling_prompt_llama = (
        f"{BOS}{B_INST} {B_SYS}\n" f"{assistant_prompt}\n" f"{E_SYS}\n\n" f"{user_prompt}\n" f"{E_INST}Results:\n"
    )
    output_list = []
    for item in chunk_of_text:
        text_item = item["fields"]["content"]
        prompt_reformed = labeling_prompt_llama.replace("{query}", query).replace("{snippet}", text_item)
        output = generate(prompt_reformed)
        output_trimmed = output.split("}")[0] + "}"
        output_dict = json.loads(output_trimmed)
        item["fields"]["relevance"] = output_dict
        # print("Test Dict ", item["fields"]["relevance"])
    return chunk_of_text


@app.route("/search", methods=["POST"])
def flask_search():
    data = request.json
    query_text = data.get("text", "").strip()
    index_name = data.get("index_name", "abstract")
    method_name = data.get("method_name", "symmetric")
    model_name = data.get("model_name", "bge")
    n_answers = data.get("n_answers", 10)
    environment_name = data.get("environment_name", "test")
    # model_selected = models[model_name]
    # meta_selected = metas[index_name]
    # index_selected = indexs[index_name][model_name]

    acquired = semaphore.acquire(blocking=True, timeout=10)
    if not acquired:
        abort(429, "Too Many Requests")
    try:
        # query_text = "How is the mechanism by which enzymes serve as biological catalyst"
        # Retrieve Step
        start_time = time.time()
        # Tokenize the string into words
        words = word_tokenize(query_text)

        # Get the list of stop words in English
        stop_words = set(stopwords.words('english'))
        # Filter out stop words
        filtered_sentence = [word for word in words if word.lower() not in stop_words]

        # Join the words back into a string
        result_query = ' '.join(filtered_sentence)
        retrieved_text = retrieve_step(result_query, n_answers , environment_name , dense=False)
        end_time = time.time()
        print("retrieve step", end_time - start_time)
        #print(retrieved_text)
        # Post - Relevance Step
        #Check the number of documentes retrieved
        totalCount =  retrieved_text["root"]["fields"]["totalCount"]
        if totalCount == 0:
            result = [{
                "error" : "Sorry, Try again with other query or question"
            }]
        else:
            result = retrieved_text["root"]["children"] 
        # Let's Delete the first result
        # result = result[1:]
        # chunk_of_text = [ text["fields"]["content"] for text in result]

        # # Relevance Step
        # start_time = time.time()
        # result = relevance_step(query_text, result)
        # end_time = time.time()
        # print("Relevance time", end_time - start_time)
        #     top_score, top_idx = top_score[0].tolist(), top_idx[0].tolist()

        #     # for doc in result:
        #     #     print(doc["similarity"])
        #     ##Answer
        #     filtered_result = [item for item in result if len(item["text"]) >= 230]
        #     # If the length of filtered_result is greater than 10, trim it
        #     trim = n_answers - 20
        #     if len(filtered_result) >= trim:
        #         filtered_result = filtered_result[:trim]
        #     # If you want to modify the 'result' in-place, you can use:
        #     result[:] = filtered_result
        #     print(len(filtered_result))
        #     print(len(result))
        #     question = query_text
        #     ANSWER_INPUT = f"""
        #     You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        #     If you don't have enough context to answer the question, just say that It doesn't enough context.
        #     Generate an answer to the user's question based on the context. Use two sentences maximum and keep the answer concise.

        #     Question: {question}

        #     Context: {result}

        #     Include as much information as possible in the answer. Reference the relevant context results urls as markdown links.These links should be taken by the DOI present in the context retrieved.
        # """
        #     completion = openai.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #     {"role": "user", "content": ANSWER_INPUT}
        #     ],
        #     temperature=0.5
        #     )
        #     openQA = {
        #         "response"  : completion.choices[0].message.content
        #     }
        #     print(completion.choices[0].message.content )
        #     result.append(openQA)
        #     print(result)
        #print(result)
        return jsonify(result=result)
    finally:
        semaphore.release()


@app.route("/gen", methods=["POST"])
def flask_gen():
    data = request.json
    question = data.get("question", "").strip()
    past_result = data.get("past_result", "")
    ##Filtered results
    context = []
    for item in past_result:
        item = item["fields"]
        print(item["relevance"]["M"])
        if item["relevance"]["M"] >= 2:
            data_chat = {"text": item["content"], "source": f"Link: http://doi.org/{item['doi']}"}
            context.append(data_chat)

    #         ##Answer
    #     filtered_result = [item for item in result if len(item["text"]) >= 230]
    #     # If the length of filtered_result is greater than 10, trim it
    #     trim = n_answers - 20
    #     if len(filtered_result) >= trim:
    #         filtered_result = filtered_result[:trim]
    #     # If you want to modify the 'result' in-place, you can use:
    #     result[:] = filtered_result
    #     print(len(filtered_result))
    #     print(len(result))
    #     question = query_text
    ANSWER_INPUT = f"""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't have enough context to answer the question, just say that It doesn't enough context. 
    Generate an answer to the user's question based on the context validating step by step each context. 
    
    Question: {question} 

    Context: {context} 

    Include as much information as possible in the answer. Reference the relevant context results urls as markdown links.These links should be taken by the DOI present in the context retrieved.
"""
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": ANSWER_INPUT}], temperature=0.5
    )
    openQA = {"response": completion.choices[0].message.content}
    print(completion.choices[0].message.content)

    resultgen = {"onix": completion.choices[0].message.content}
    return jsonify(result=resultgen)


@app.route("/dense", methods=["POST"])
def flask_search_dense():
    global models, metas, indexs

    data = request.json
    
    #method_name = data.get("method_name", "symmetric")
    query_text = data.get("text", "").strip()
    n_answers = data.get("n_answers", 10)
    environment_name = data.get("environment_name", "test")
    model_selected = models["bge"]
    meta_selected = metas["sentence"]
    index_selected = indexs["sentence"]["bge"]
    

    method_name = "asymmetric"
    acquired = semaphore.acquire(blocking=True, timeout=10)
    if not acquired:
        abort(429, "Too Many Requests")
    try:
        if environment_name == "test":
            if method_name == "asymmetric":
                features = model_selected.batch_query_tokenize((query_text,))
            else:
                features = model_selected.batch_doc_tokenize((query_text,))
            start_time = time.time()
            query_emb, seq_len = model_selected.batch_encode(features)
            end_time = time.time()
            print('encode latency', end_time - start_time)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            query_emb = query_emb.numpy()
            start_time = time.time()
            top_score, top_idx = index_selected.search(query_emb, n_answers)
            end_time = time.time()
            print('search time', end_time - start_time)
            top_score, top_idx = top_score[0].tolist(), top_idx[0].tolist()

            result = []
            for i in range(len(top_score)):
                doc = meta_selected[top_idx[i]]
                doc["similarity"] = 1 - top_score[i] / 2
                result.append(doc)
        # for doc in result:
        #     print(doc["similarity"])
        
        if environment_name == "production":
            retrieved_text = retrieve_step(query_text, n_answers , environment_name , dense=True) 
            totalCount =  retrieved_text["root"]["fields"]["totalCount"]
            if totalCount == 0:
                result = [{
                    "error" : "Sorry, Try again with other query or question"
                }]
            else:
                result = retrieved_text["root"]["children"] 
        return jsonify(result=result)
    finally:
        semaphore.release()



@app.route("/reranking", methods=["POST"])
def flask_reranking():
    global models, metas, indexs

    data = request.json
    
    #method_name = data.get("method_name", "symmetric")
    query_text = data.get("text", "").strip()
    n_answers = data.get("n_answers", 10)
    environment_name = data.get("environment_name", "test")
    model_selected = models["bge"]
    meta_selected = metas["sentence"]
    index_selected = indexs["sentence"]["bge"]
   
    method_name = "asymmetric"
    acquired = semaphore.acquire(blocking=True, timeout=10)
    if not acquired:
        abort(429, "Too Many Requests")
    try:
        if environment_name == "test":
            if method_name == "asymmetric":
                features = model_selected.batch_query_tokenize((query_text,))
            else:
                features = model_selected.batch_doc_tokenize((query_text,))
            start_time = time.time()
            query_emb, seq_len = model_selected.batch_encode(features)
            end_time = time.time()
            print('encode latency', end_time - start_time)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            query_emb = query_emb.numpy()
            start_time = time.time()
            top_score, top_idx = index_selected.search(query_emb, n_answers)
            end_time = time.time()
            print('search time', end_time - start_time)
            top_score, top_idx = top_score[0].tolist(), top_idx[0].tolist()

            result = []
            for i in range(len(top_score)):
                doc = meta_selected[top_idx[i]]
                doc["similarity"] = 1 - top_score[i] / 2
                result.append(doc)
        # for doc in result:
        #     print(doc["similarity"])
        
        if environment_name == "production":
            retrieved_text = retrieve_step(query_text, n_answers , environment_name , dense=True) 
            totalCount =  retrieved_text["root"]["fields"]["totalCount"]
            if totalCount == 0:
                result = [{
                    "error" : "Sorry, Try again with other query or question"
                }]
            else:
                result = retrieved_text["root"]["children"] 
        return jsonify(result=result)
    finally:
        semaphore.release()


if __name__ == "__main__":
    initialize_app()
    app.run(port=5001, debug=False)
