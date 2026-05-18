from argparse import ArgumentParser
import gc
import glob
import json
import os
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from transformers import LlamaModel, LlamaTokenizerFast, BertModel, BertTokenizerFast
from optimum.bettertransformer import BetterTransformer
from collections import deque
from datasets import load_dataset


def batch_to_device(batch, target_device: torch.device):
    cpu_device = torch.device("cpu")
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            if batch[key].device != cpu_device:
                batch[key] = batch[key].pin_memory()
            batch[key] = batch[key].to(target_device, non_blocking=True)
    return batch


class RepLLaMA:
    def __init__(self, device: str):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "meta-llama/Llama-2-7b-hf", cache_dir="./llm", token="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"
        )
        self.tokenizer.add_eos_token = True
        # add eos token automatically since repllama ends with </s>

        # setup special tokens
        # 1 <s>
        # 2 </s>
        # 0 <unk>
        # llama2 original code use -1 but hf does not support -1
        # in theory unk will never be used since it add 0x00-0xFF to vocab
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # setup padding
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.tokenizer.model_max_length = 2048

        if device == "cpu":
            self.model = LlamaModel.from_pretrained(
                "llm/repllama-v1-7b-lora-passage", torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
        else:
            # direct load to cuda greatly reduce initialzation time
            # checkpoint is float32, use float16 since it can run on V100
            # bfloat16 trec-covid ndcg@10:0.8412
            # float16 trec-covid ndcg@10:0.8425
            # int8 trec-covid ndcg@10:0.8456
            self.model = LlamaModel.from_pretrained(
                "llm/repllama-v1-7b-lora-passage",
                torch_dtype=torch.float16,
                device_map=0,
            )
        self.model = BetterTransformer.transform(self.model)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # tensor([ 0.0165,  0.0180,  0.0317,  ..., -0.0164,  0.0056,  0.0163])
        print(self.model.layers[0].mlp.gate_proj.weight[0])
        self.model.eval()
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def batch_to_device(batch, target_device: torch.device):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(target_device)
        return batch

    def batch_doc_tokenize(self, text_list: List[Union[str, Tuple[str, str]]]):
        passage_input = []
        for passage in text_list:
            if type(passage) == str:
                passage_input.append(f"passage: {passage.strip()}")
            elif type(passage) == tuple:
                passage_input.append(f"passage: {passage[0].strip()} {passage[1].strip()}")
        return self.batch_tokenize(passage_input)

    def batch_query_tokenize(self, text_list: List[str]):
        query_input = [f"query: {query.strip()}" for query in text_list]
        return self.batch_tokenize(query_input)

    def batch_tokenize(self, text_list: List[str]):
        batch_input = self.tokenizer(text_list, padding=True, truncation=True, max_length=2048, return_tensors="pt")
        bsz = batch_input["input_ids"].size(0)
        seq_len = torch.eq(batch_input["input_ids"], self.tokenizer.eos_token_id).long().argmax(-1)
        assert torch.all(batch_input["input_ids"][torch.arange(bsz), seq_len] == self.tokenizer.eos_token_id)
        return batch_to_device(batch_input, self.model.device)

    def batch_encode(self, batch_input: Dict):
        with torch.inference_mode():
            # compute query embedding
            outputs = self.model(input_ids=batch_input["input_ids"], output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            batch_size, seq_len = batch_input["input_ids"].size()
            eos_pos = torch.eq(batch_input["input_ids"], self.tokenizer.eos_token_id).long().argmax(-1)
            embedding = hidden_states[torch.arange(batch_size, device=hidden_states.device), eos_pos, :]
            # embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.clone().detach().to("cpu", non_blocking=True), seq_len


class BGE:
    def __init__(self, device: str):
        self.tokenizer = BertTokenizerFast.from_pretrained("BAAI/bge-large-en-v1.5", cache_dir="llm")
        if device == "cpu":
            self.model = BertModel.from_pretrained("BAAI/bge-large-en-v1.5", cache_dir="llm")
        else:
            self.model = BertModel.from_pretrained(
                "BAAI/bge-large-en-v1.5", torch_dtype=torch.float16, device_map=device, cache_dir="llm"
            )
        # BetterTransformer does not work with attention mask for training
        self.model = BetterTransformer.transform(self.model)
        self.model.eval()
        torch.cuda.empty_cache()
        gc.collect()

    def batch_doc_tokenize(self, text_list: List[Union[str, Tuple[str, str]]]):
        passage_input = []
        for passage in text_list:
            if type(passage) == str:
                passage_input.append(f"{passage.strip()}")
            elif type(passage) == tuple:
                passage_input.append(f"{passage[0].strip()} {passage[1].strip()}")
        return self.batch_tokenize(passage_input)

    def batch_query_tokenize(self, text_list: List[str]):
        query_input = [
            f"Represent this sentence for searching relevant passages: {query.strip()}" for query in text_list
        ]
        return self.batch_tokenize(query_input)

    def batch_tokenize(self, text_list: List[str]):
        batch_input = self.tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return batch_to_device(batch_input, self.model.device)

    def batch_encode(self, batch_input: Dict):
        with torch.inference_mode():
            # compute query embedding
            outputs = self.model(**batch_input)
            batch_size, seq_len = batch_input["input_ids"].size()
            # Perform pooling. In this case, cls pooling.
            embedding = outputs[0][:, 0]
            # embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.clone().detach().to("cpu", non_blocking=True), seq_len


class GTE:
    def __init__(self, device: str):
        self.tokenizer = BertTokenizerFast.from_pretrained("thenlper/gte-large", cache_dir="llm")
        if device == "cpu":
            self.model = BertModel.from_pretrained("thenlper/gte-large", cache_dir="llm")
        else:
            self.model = BertModel.from_pretrained(
                "thenlper/gte-large", torch_dtype=torch.float16, device_map=device, cache_dir="llm"
            )
        self.model = BetterTransformer.transform(self.model)
        self.model.eval()
        torch.cuda.empty_cache()
        gc.collect()

    def batch_doc_tokenize(self, text_list: List[Union[str, Tuple[str, str]]]):
        passage_input = []
        for passage in text_list:
            if type(passage) == str:
                passage_input.append(f"{passage.strip()}")
            elif type(passage) == tuple:
                passage_input.append(f"{passage[0].strip()} {passage[1].strip()}")
        return self.batch_tokenize(passage_input)

    def batch_query_tokenize(self, text_list: List[str]):
        query_input = [f"{query.strip()}" for query in text_list]
        return self.batch_tokenize(query_input)

    def batch_tokenize(self, text_list: List[str]):
        batch_input = self.tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return batch_to_device(batch_input, self.model.device)

    def batch_encode(self, batch_input: Dict):
        with torch.inference_mode():
            # compute query embedding
            outputs = self.model(**batch_input)
            attention_mask = batch_input["attention_mask"]
            batch_size, seq_len = batch_input["input_ids"].size()
            # Perform pooling. In this case, avg pooling.
            last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embedding = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            # embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.clone().detach().to("cpu", non_blocking=True), seq_len


def encode(
    model: Union[RepLLaMA, BGE, GTE],
    texts: List[Union[str, Tuple[str, str]]],
    text_type: str,
    batch_size: int = 32,
) -> np.ndarray:
    all_embeddings = []
    if text_type == "doc":
        tokenize_fn = model.batch_doc_tokenize
    elif text_type == "query":
        tokenize_fn = model.batch_query_tokenize
    else:
        raise RuntimeError("unknown text type")

    if type(texts[0]) == str:
        # by default it uses quicksort which is not stable
        length_sorted_idx = np.argsort([len(text) for text in texts])
    else:
        length_sorted_idx = np.argsort([len(title) + len(text) for title, text in texts])
    texts_sorted = [texts[idx] for idx in length_sorted_idx]
    pbar = trange(0, len(texts), batch_size, desc="Batches")
    for start_index in pbar:
        texts_batch = texts_sorted[start_index : start_index + batch_size]
        features = tokenize_fn(texts_batch)
        embeddings, seq_len = model.batch_encode(features)
        pbar.set_description(f"Processing {seq_len} tokens")
        all_embeddings.append(embeddings)
    # otherwise embedding may not finish transfer
    torch.cuda.synchronize()
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    return torch.stack(all_embeddings)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", choices=["encode", "merge"])
    parser.add_argument("--input_fn")
    parser.add_argument("--output_dir")
    parser.add_argument("--model", choices=["bge", "gte", "repllama"])
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--shard_num", type=int, default=1)
    # parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    # CUDA_VISIBLE_DEVICES must be set at python running
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    if args.task == "merge":
        fps = list(glob.glob(f"{args.output_dir}/{args.model}/corpus_embedding_*"))
        assert len(fps) == args.shard_num, "find incorrect num of shard result"
        corpus_emb_list = {}
        for bin_file in fps:
            prefix, suffix = os.path.basename(bin_file).rsplit("_", 1)
            shard_id, suffix = suffix.split(".")
            shard_id = int(shard_id)
            shard_emb = torch.load(bin_file, map_location=torch.device("cpu"))
            print(type(shard_emb), type(shard_emb[0]), len(shard_emb))
            corpus_emb_list = torch.stack(shard_emb)

        shard_ids = sorted(list(corpus_emb_list.keys()))
        corpus_emb_list = [corpus_emb_list[shard_id] for shard_id in shard_ids]

        corpus_emb = torch.concat(corpus_emb_list, dim=0)
        print(corpus_emb.shape)
        torch.save(corpus_emb, f"{args.output_dir}/{args.model}/corpus_embedding.pth")
    else:
        data = load_dataset(
            "json",
            data_files=args.input_fn,
            cache_dir=f"{args.output_dir}/.cache"
        )
        full_data = data["train"]
        #papers = full_data["segment"]
        papers = full_data["content"]
        print(len(papers), "papers loaded")

        if args.model == "bge":
            model = BGE(f"cuda")
            batch_size = 32
        elif args.model == "gte":
            model = GTE(f"cuda")
            batch_size = 32
        elif args.model == "repllama":
            model = RepLLaMA(f"cuda")
            batch_size = 4
        else:
            raise RuntimeError("unknown model")

        corpus_embeddings = encode(model, papers, "doc", batch_size)

        torch.save(corpus_embeddings, f"{args.output_dir}/{args.model}/corpus_embedding_{args.shard_id}.pth")