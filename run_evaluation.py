THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',

    'mrtydi-ar': 'mrtydi-v1.1-arabic',
    'mrtydi-bn': 'mrtydi-v1.1-bengali',
    'mrtydi-fi': 'mrtydi-v1.1-finnish',
    'mrtydi-id': 'mrtydi-v1.1-indonesian',
    'mrtydi-ja': 'mrtydi-v1.1-japanese',
    'mrtydi-ko': 'mrtydi-v1.1-korean',
    'mrtydi-ru': 'mrtydi-v1.1-russian',
    'mrtydi-sw': 'mrtydi-v1.1-swahili',
    'mrtydi-te': 'mrtydi-v1.1-telugu',
    'mrtydi-th': 'mrtydi-v1.1-thai',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',

    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',

}

from rank_gpt import run_retriever, sliding_windows, write_eval_file , receive_permutation_v2
from pyserini.search import get_topics, get_qrels
from pyserini.search.lucene import LuceneSearcher
from datetime import datetime
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import time
import tempfile
import os
import json
import shutil

def reranking_cohere(user_query:str, documents:List[str], top_n:int) -> List[int]:

    import cohere

    co = cohere.ClientV2(
        "bZfph2epGHhjVGrZAUyzznqf2Aa6yqOW16migPRa"
    )  # Get your free API key here: https://dashboard.cohere.com/api-keys
    results = co.rerank(model="rerank-v3.5", query=user_query, documents=documents, top_n=top_n ,return_documents=True)

    print("Cohore Rerank")

    return [result.index for result in results.results]


openai_key = os.environ.get("OPENAI_API_KEY", None)

#for data in ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']:
for data in ['dl19']:
    model_list = ["deepseek-r1:14b", "deepseek-r1:8b", "llama3.1" , "phi4" , "gpt-4o-mini", "cohere","deepseek-r1:7b"]
    model_name = "deepseek-r1:7b"
    print('#' * 20)
    print(f'Evaluation on {data} with {model_name}')
    print('#' * 20)

    
    # Retrieve passages using pyserini BM25.
    try:
        searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
        topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
        qrels = get_qrels(THE_TOPICS[data])
        rank_results = run_retriever(topics, searcher, qrels, k=100)
    except:
        print(f'Failed to retrieve passages for {data}')
        continue
    
    # Run sliding window permutation generation
    new_results = []
    intx = 0
    for item in tqdm(rank_results):
        if model_name != "cohere":        
            new_item = sliding_windows(item, rank_start=0, rank_end=100, window_size=20, step=10,
                                    model_name=model_name, api_key="openai_key")
        else:        
            user_query = item['query']
            documents = [hit['content'] for hit in item['hits']]
            rank_list = reranking_cohere(user_query, documents , 100)
            new_item = receive_permutation_v2(item,rank_list)
            intx  += 1
            if intx %9 == 0 : time.sleep(60)
        new_results.append(new_item)
    
      
    # Evaluate nDCG@10
    from trec_eval import EvalFunction

    # Create an empty text file to write results, and pass the name to eval
    #temp_file = tempfile.NamedTemporaryFile(delete=False).name
    save_dir = "/data_hdd/damian/pyserini"
    os.makedirs(save_dir, exist_ok=True)
    # Generate a unique filename using the current date and time
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    temp_file = os.path.join(save_dir, f"results_{model_name}_{timestamp}.txt")
    

    EvalFunction.write_file(new_results, temp_file)
    results =EvalFunction.main(THE_TOPICS[data], temp_file)
    
    log_file = os.path.join(save_dir, f"evaluation_{model_name}_{timestamp}.json")
    with open(log_file, "w") as f:
        json.dump(results, f, indent=4)
    

# for data in ['mrtydi-ar', 'mrtydi-bn', 'mrtydi-fi', 'mrtydi-id', 'mrtydi-ja', 'mrtydi-ko', 'mrtydi-ru', 'mrtydi-sw', 'mrtydi-te', 'mrtydi-th']:
#     print('#' * 20)
#     print(f'Evaluation on {data}')
#     print('#' * 20)

#     # Retrieve passages using pyserini BM25.
#     try:
#         searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
#         topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
#         qrels = get_qrels(THE_TOPICS[data])
#         rank_results = run_retriever(topics, searcher, qrels, k=100)
#         rank_results = rank_results[:100]

#     except:
#         print(f'Failed to retrieve passages for {data}')
#         continue

#     # Run sliding window permutation generation
#     new_results = []
#     for item in tqdm(rank_results):
#         new_item = sliding_windows(item, rank_start=0, rank_end=100, window_size=20, step=10,
#                                    model_name='gpt-3.5-turbo', api_key=openai_key)
#         new_results.append(new_item)

#     # Evaluate nDCG@10
#     from trec_eval import EvalFunction

#     temp_file = tempfile.NamedTemporaryFile(delete=False).name
#     EvalFunction.write_file(new_results, temp_file)
#     EvalFunction.main(THE_TOPICS[data], temp_file)