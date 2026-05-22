import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import tqdm
import json
import random
import math
import numpy as np



class EmbeddingDataset(Dataset):
    def __init__(
            self,
            train_data_path,
            tokenizer,
            neg_nums,
            query_max_len=128,
            passage_max_len=512,
    ):

        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

        self.neg_nums = neg_nums
        self.train_data = self.read_train_data(train_data_path)
        train_data_keys_num = len(self.train_data[0].keys())
        train_data_keys_num = train_data_keys_num - 1 if 'prompt_for_query' in self.train_data[0] else train_data_keys_num
        if train_data_keys_num == 2:
            self.data_type = 'pair'
        elif train_data_keys_num == 3 and 'neg' in self.train_data[0]:
            self.data_type = 'triplet'
        elif train_data_keys_num == 3 and 'score' in self.train_data[0]:
            self.data_type = 'pair_score'
        self.tokenizer = tokenizer
        self.data_type_2_collate_fn = {
            "pair": self.pair_collate_fn,
            "triplet": self.triplet_collate_fn,
            "pair_score": self.pair_score_collate_fn
        }
        self.collate_fn = self.data_type_2_collate_fn[self.data_type]

    def read_train_data(self, train_data_path):
        train_data = []
        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                if "prompt_for_query" in data_dic and data_dic["prompt_for_query"]:
                    data_dic['query'] = data_dic["prompt_for_query"] + data_dic['query']
                if 'pos' in data_dic and 'neg' not in data_dic:
                    for i, text_pos in enumerate(data_dic['pos']):
                        temp_dic = {}
                        temp_dic['query'] = data_dic['query']
                        temp_dic['pos'] = text_pos
                        if 'scores' in data_dic:
                            temp_dic['score'] = data_dic['scores'][i]
                        train_data.append(temp_dic)
                elif 'pos' in data_dic and 'neg' in data_dic:
                    for text_pos in data_dic['pos']:
                        temp_dic = {}
                        temp_dic['query'] = data_dic['query']
                        temp_dic['pos'] = text_pos
                        if len(data_dic['neg']) < self.neg_nums:
                            num = math.ceil(self.neg_nums / len(data_dic['neg']))
                            temp_dic['neg'] = random.sample(data_dic['neg'] * num, self.neg_nums)
                        else:
                            temp_dic['neg'] = random.sample(data_dic['neg'], self.neg_nums)

                        train_data.append(temp_dic)

        return train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def triplet_collate_fn(self, batch):
        all_querys = []
        all_pos_docs = []
        all_neg_docs = []
        for item in batch:
            all_querys.append(item['query'])
            all_pos_docs.append(item['pos'])
            all_neg_docs.extend(item['neg'])
        all_query_tokens = self.tokenizer(all_querys, padding='max_length', truncation=True,
                                          max_length=self.query_max_len, return_tensors='pt')
        all_pos_doc_tokens = self.tokenizer(all_pos_docs, padding='max_length', truncation=True,
                                            max_length=self.passage_max_len, return_tensors='pt')
        all_neg_doc_tokens = self.tokenizer(all_neg_docs, padding='max_length', truncation=True,
                                            max_length=self.passage_max_len, return_tensors='pt')
        tokens_batch = {}
        tokens_batch['query_input_ids'] = all_query_tokens['input_ids']
        tokens_batch['query_attention_mask'] = all_query_tokens['attention_mask']
        tokens_batch['pos_doc_input_ids'] = all_pos_doc_tokens['input_ids']
        tokens_batch['pos_doc_attention_mask'] = all_pos_doc_tokens['attention_mask']
        tokens_batch['neg_doc_input_ids'] = all_neg_doc_tokens['input_ids']
        tokens_batch['neg_doc_attention_mask'] = all_neg_doc_tokens['attention_mask']
        return tokens_batch

    def pair_collate_fn(self, batch):

        all_querys = []
        all_pos_docs = []
        for item in batch:
            all_querys.append(item['query'])
            all_pos_docs.append(item['pos'])
        all_query_tokens = self.tokenizer(all_querys, padding='max_length', truncation=True,
                                          max_length=self.query_max_len, return_tensors='pt')
        all_pos_doc_tokens = self.tokenizer(all_pos_docs, padding='max_length', truncation=True,
                                            max_length=self.passage_max_len, return_tensors='pt')
        tokens_batch = {}
        tokens_batch['query_input_ids'] = all_query_tokens['input_ids']
        tokens_batch['query_attention_mask'] = all_query_tokens['attention_mask']
        tokens_batch['pos_doc_input_ids'] = all_pos_doc_tokens['input_ids']
        tokens_batch['pos_doc_attention_mask'] = all_pos_doc_tokens['attention_mask']
        return tokens_batch

    def pair_score_collate_fn(self, batch):

        all_querys = []
        all_pos_docs = []
        scores = []
        for item in batch:
            all_querys.append(item['query'])
            all_pos_docs.append(item['pos'])
            scores.append(item['score'])
        all_query_tokens = self.tokenizer(all_querys, padding='max_length', truncation=True,
                                          max_length=self.query_max_len, return_tensors='pt')
        all_pos_doc_tokens = self.tokenizer(all_pos_docs, padding='max_length', truncation=True,
                                            max_length=self.passage_max_len, return_tensors='pt')
        tokens_batch = {}
        tokens_batch['query_input_ids'] = all_query_tokens['input_ids']
        tokens_batch['query_attention_mask'] = all_query_tokens['attention_mask']
        tokens_batch['pos_doc_input_ids'] = all_pos_doc_tokens['input_ids']
        tokens_batch['pos_doc_attention_mask'] = all_pos_doc_tokens['attention_mask']
        tokens_batch['scores'] = torch.tensor(scores)
        return tokens_batch

class EmbeddingDistillDataset(Dataset):
    def __init__(
        self,
        train_data_path,
        train_dataset_vec_path,
        tokenizer,
        teacher_embedding_dim,
        query_max_len=512,
        data_type="distill",
    ):
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.train_data_text = self.read_train_data(train_data_path)

        print(len(self.train_data_text))
        self.train_data_embedding_mmap = np.memmap(train_dataset_vec_path, 
            dtype='float32', mode='r', shape=(len(self.train_data_text), teacher_embedding_dim))
        
        assert self.train_data_embedding_mmap[len(self.train_data_text)-1] is not None

        self.collate_fn = self.collate_fn
        self.data_type = data_type

    def read_train_data(self, train_data_path):
        train_data = []
        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                if "prompt_for_query" in data_dic and data_dic["prompt_for_query"]:
                    data_dic['query'] = data_dic["prompt_for_query"] + data_dic['query']
                temp_dic = {}
                temp_dic['query'] = data_dic['query']
                train_data.append(temp_dic)

        return train_data

    def __len__(self):
        return len(self.train_data_text)

    def __getitem__(self, idx):
        embedding = self.train_data_embedding_mmap[idx].tolist()
        query = self.train_data_text[idx]['query']
        return {'query': query, 'embedding': embedding}

    def collate_fn(self, batch):

        all_querys = []
        all_teacher_embeddings = []
        for item in batch:
            all_querys.append(item['query'])
            all_teacher_embeddings.append(item['embedding'])
        all_query_tokens = self.tokenizer(all_querys, padding='longest', truncation=True,
                                          max_length=self.query_max_len, return_tensors='pt')
        all_teacher_embeddings = torch.tensor(all_teacher_embeddings)
        tokens_batch = {}
        tokens_batch['query_input_ids'] = all_query_tokens['input_ids']
        tokens_batch['query_attention_mask'] = all_query_tokens['attention_mask']
        tokens_batch['teacher_embeddings'] = all_teacher_embeddings



        return tokens_batch

def test_EmbeddingDataset():
    train_data_path = '../../../example_data/t2rank_100.jsonl'
    model_name_or_path = 'hfl/chinese-roberta-wwm-ext'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset = EmbeddingDataset(train_data_path, tokenizer, 15)
    print('using ', dataset.data_type)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            collate_fn=dataset.pair_collate_fn if dataset.data_type == 'pair' else dataset.triplet_collate_fn,
                            )
    print(len(dataloader))
    for batch in tqdm.tqdm(dataloader):
        print(batch['query_input_ids'].size())
        print(batch['pos_doc_attention_mask'].size())
        print(batch['neg_doc_attention_mask'].size())
        break

def test_EmbeddingDistillDataset():
    train_data_path = '../../../example_data/t2rank_100.jsonl.text.jsonl'
    train_dataset_vec_path="../../../example_data/t2rank_100.embedding.conan.xiaobu.mmap"
    model_name_or_path = 'BAAI/bge-base-zh-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset = EmbeddingDistillDataset(train_data_path,train_dataset_vec_path, tokenizer,teacher_embedding_dim=1792*2)
    dataloader = DataLoader(dataset,
                            batch_size=512,
                            shuffle=False,
                            collate_fn=dataset.collate_fn,
                            )
    print(len(dataloader))
    for batch in tqdm.tqdm(dataloader):
        print(batch['query_input_ids'])
        print(tokenizer.decode(batch['query_input_ids'][0]))
        print(batch['teacher_embeddings'])
        break

if __name__ == "__main__":
    test_EmbeddingDistillDataset()
