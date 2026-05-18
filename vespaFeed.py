import re
import pymysql.cursors
from pymongo.mongo_client import MongoClient
import ciso8601
import pandas as pd
import datetime
import ftfy
import requests
import textacy.preprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import json
import logging
import argparse
from MongoExtract import DataBaseFetcher, ParagraphExtractor

class vespaFileExtractor:

  
    def __init__(self, publisher, normalize=True):
        self.publisher = publisher
        self.normalize = normalize
        self.fetcher = DataBaseFetcher()
        self.extractor = ParagraphExtractor(normalize)
        # self.doi_parse = self.fetcher.parsed_doi()

    def __call__(self, doi,publisher):
        item = self.fetcher.retriveDataMongo(doi, publisher)
        if item is None:
            return None

        publisher, title, journal = item["publisher"], item["title"], item["journal"]
        timestamp, authors, citation = self.fetcher.crossrefData(doi)

        result = []
        idx = 0
        for is_abstract, path, contents in self.extractor(item):
            doc_type = "plastics"
            doc = {
                "put": f"id:{doc_type}:{doc_type}::{doi}/{idx + 1}",
                "fields": {
                    "doi": doi,
                    "publisher": publisher,
                    "title": title,
                    "journal": journal,
                    # "timestamp": timestamp,
                    # "authors": authors,
                    # "citation": citation,
                    "content": '\n'.join(contents),
                    # "titleSection": '|'.join(path),
                    # "numParagraph": idx + 1,
                    # "isAbstract": is_abstract
                }
            }
            result.append(doc)
            idx = idx + 1
        return result


def feed_vespa_api(doc):
    header = {'Content-type': 'application/json'}
    doc_type = "article"
    doi_paraid = doc["put"].split('::')[-1]
    url = f"http://localhost:8082/document/v1/{doc_type}/{doc_type}/docid/{doi_paraid}"
    r = requests.post(url, json={'fields': doc['fields']}, headers=header)

    if r.status_code != 200:
        raise RuntimeError("Code:{0} Text:{1}".format(r.status_code, r.text))


def multithread_call(fn, arg_list, batch_size=10000, num_worker=5):
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        buf = []
        submit = 0
        success = 0
        for arg in arg_list:
            buf.append(executor.submit(fn, *arg))
            submit += 1
            if len(buf) >= batch_size:
                logging.info(f'submit {submit}')
                for future in tqdm(as_completed(buf)):
                    try:
                        future.result()
                        success += 1
                    except Exception as e:
                        logging.info(str(e))
                logging.info(f'finish {success}')
                buf = []
        if len(buf) > 0:
            for future in tqdm(as_completed(buf)):
                try:
                    future.result()
                    success += 1
                except Exception as e:
                    logging.info(str(e))
            logging.info(f'finish {success}')
        logging.info(f'submit {submit} finish {success}')


if __name__ == "__main__":
    #Initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('--pub',
                        choices=['Elsevier', 'Nature', 'RSC', 'Springer' , 'ACS', 'Wiley'],
                        default='Wiley',
                        help='Name of publisher:  Elsevier  Nature  RSC  Springer ')
    parser.add_argument('--opt',
                        choices=['json', 'post'],
                        default='json',
                        help='generate json or directly put into vespa through post')
    parser.add_argument('--debug', action='store_true')
    arg = parser.parse_args()

    if arg.opt == "json":
        logging.basicConfig(level=logging.INFO)
        #out_file = os.path.join(os.path.dirname(__file__), f"json_output/{arg.pub}.json")
        out_file = os.path.join("/data_hdd/damian/fulltext", f"{arg.pub}.json")
        extractor = vespaFileExtractor(arg.pub)
        
        #Read Document and filter by the Publisher
        df = pd.read_json('doi_publisher_list.json', orient='records', lines=True)
        df_filtered = df[df['Publisher'] == arg.pub]
        doi_list = df_filtered['DOI'].tolist()

        with open(out_file, 'w') as f:
            
            # for doi in tqdm(extractor.doi_parse):
            for doi in doi_list:
                docs = extractor(doi , arg.pub)  # Retrive data
                if docs:
                    for doc in docs:
                        f.write(json.dumps(doc) + '\n')

    elif arg.opt == "post":
        logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), f'post_log/{arg.pub}.log'),
                            level=logging.INFO,
                            filemode='w')
        extractor = vespaFileExtractor(arg.pub)

        def extract_post(doi):
            try:
                docs = extractor(doi)  # Retrive data
                if docs:
                    for doc in docs:
                        feed_vespa_api(doc)
            except Exception as e:
                raise Exception(f'{doi} raise {repr(e)}')

        if not arg.debug:
            multithread_call(extract_post, extractor.doi_parse)
        else:
            for doi in tqdm(extractor.doi_parse):
                extract_post(doi)
