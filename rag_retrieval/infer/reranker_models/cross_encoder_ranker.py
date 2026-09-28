

from typing import Union, List, Optional, Tuple
from .ranker import BaseRanker
from .result import RankedResults, Result
from copy import deepcopy
from .utils import get_device,get_dtype,vprint
import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CorssEncoderRanker(BaseRanker):
    def __init__(self, 
        model_name_or_path: str, 
        dtype: str = None,
        device: str = None,
        verbose: int = 1,
    ):
        self.verbose = verbose
        self.model_name_or_path = model_name_or_path
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, device=self.device, verbose=self.verbose)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, torch_dtype=self.dtype
        ).to(self.device)

        vprint(f"Loaded model {self.model_name_or_path}", self.verbose)

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # if device is not None and device.startswith('cuda:'):
        #     self.num_gpus = torch.cuda.device_count()
        #     if self.num_gpus > 1:
        #         vprint(f"----------using {self.num_gpus}*GPUs----------",self.verbose)
        #         self.model = torch.nn.DataParallel(self.model)
        # else:
        #     self.num_gpus = 1
    
    @torch.no_grad()
    def compute_score(self, 
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 256,
        max_length: int = 512,
        normalize: bool = False,
        enable_tqdm: bool = True,
    ):
        
        # batch inference
        # if self.num_gpus > 1:
        #     batch_size = batch_size * self.num_gpus

        all_scores = []
        for start_index in tqdm.tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores",
                                    disable=not enable_tqdm):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)

            scores = self.model(**inputs).logits.view(-1, ).float()
            if 'bce' in self.model_name_or_path or normalize:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().numpy().tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores

    @torch.no_grad()
    def rerank(self, 
        query: str, 
        docs: Union[List[str], str] = None,
        batch_size: int = 256,
        max_length: int = 512,
        normalize: bool = False,
        long_doc_process_strategy: str = "max_score_slice",#['max_score_slice','max_length_truncation']
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):  
        if isinstance(docs, str):
            docs = [docs]

        # remove invalid docs
        docs = [doc[:128000] for doc in docs if isinstance(doc, str) and 0 < len(doc)]

        if query is None or len(query) == 0 or len(docs) == 0:
            return RankedResults(results=[], query=query or "", has_scores=True)
        
        vprint(f'long_doc_process_strategy is {long_doc_process_strategy}',self.verbose)
        if long_doc_process_strategy=='max_length_truncation':
            ranked_results = self.__max_length_truncation_rerank(query,docs,batch_size,max_length,normalize)
        else:
            ranked_results = self.__max_score_slice_rerank(query,docs,batch_size,max_length,normalize)

        if score_threshold is not None:
            ranked_results = ranked_results.filter_by_score(score_threshold)
        if top_k is not None and top_k > 0:
            ranked_results.results = ranked_results.results[:top_k]

        return ranked_results

    @torch.no_grad()
    def __max_length_truncation_rerank(self,
        query: str, 
        docs: Union[List[str], str] = None,
        batch_size: int = 256,
        max_length: int = 512,
        normalize: bool = False,
    ):
        doc_ids = list(range(len(docs)))
        sentence_pairs=[ [query,doc]  for doc in docs]
        all_scores = self.compute_score(sentence_pairs,batch_size,max_length,normalize=normalize,enable_tqdm=False)

        ranked_results = [
            Result(doc_id=doc_id, text=doc, score=score, rank=idx + 1)
            for idx, (doc_id, doc, score) in enumerate(
                sorted(zip(doc_ids, docs, all_scores), key=lambda x: x[2], reverse=True)
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    @torch.no_grad()
    def __max_score_slice_rerank(self,
        query: str, 
        docs: Union[List[str], str] = None,
        batch_size: int=256,
        max_length: int = 512,
        normalize: bool = False,
        overlap_tokens_length: int=80,
    ):

        doc_ids = list(range(len(docs)))
        
        # preproc of tokenization
        sentence_pairs, sentence_pairs_idxs = self.__reranker_tokenize_preproc(
            query,
            docs, 
            max_length=max_length,
            overlap_tokens_length=overlap_tokens_length,
        )

        # batch inference
        # if self.num_gpus > 1:
        #     batch_size = batch_size * self.num_gpus

        all_scores = []
        for start_index in range(0, len(sentence_pairs), batch_size):
            batch = self.tokenizer.pad(
                    sentence_pairs[start_index:start_index+batch_size],
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors="pt"
                ).to(self.device)
            #batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
            scores = self.model(**batch).logits.view(-1,).float()
            if 'bce' in self.model_name_or_path or normalize:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().numpy().tolist())

        # ranking
        merge_scores = [float("-inf") for _ in range(len(docs))]
        for idx, score in zip(sentence_pairs_idxs, all_scores):
            merge_scores[idx] = max(merge_scores[idx], score)

        ranked_results = [
            Result(doc_id=doc_id, text=doc, score=score, rank=idx + 1)
            for idx, (doc_id, doc, score) in enumerate(
                sorted(zip(doc_ids, docs, merge_scores), key=lambda x: x[2], reverse=True)
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def __reranker_tokenize_preproc(self,
        query: str, 
        docs: List[str],
        max_length: int = 512,
        overlap_tokens_length: int = 80,
    ):

        sep_id = self.tokenizer.sep_token_id
        def _merge_inputs(chunk1_raw, chunk2):
            chunk1 = deepcopy(chunk1_raw)

            #add sep
            chunk1['input_ids'].append(sep_id)
            chunk1['input_ids'].extend(chunk2['input_ids'])
            chunk1['input_ids'].append(sep_id)

            chunk1['attention_mask'].append(chunk2['attention_mask'][0])
            chunk1['attention_mask'].extend(chunk2['attention_mask'])
            chunk1['attention_mask'].append(chunk2['attention_mask'][0])

            if 'token_type_ids' in chunk1:
                token_type_ids = [1 for _ in range(len(chunk2['token_type_ids'])+2)]
                chunk1['token_type_ids'].extend(token_type_ids)
            return chunk1

        query_inputs = self.tokenizer.encode_plus(query, truncation=False, padding=False)
        max_doc_inputs_length = max_length - len(query_inputs['input_ids']) - 2
        assert max_doc_inputs_length > 100, "Your query is too long! Please make sure your query less than 400 tokens!"
        overlap_tokens_length_implt = min(overlap_tokens_length, max_doc_inputs_length//4)
        

        sentence_pairs = []
        sentence_pairs_idxs = []
        for idx, doc in enumerate(docs):
            doc_inputs = self.tokenizer.encode_plus(doc, truncation=False, padding=False, add_special_tokens=False)
            doc_inputs_length = len(doc_inputs['input_ids'])

            if doc_inputs_length <= max_doc_inputs_length:
                qp_merge_inputs = _merge_inputs(query_inputs, doc_inputs)
                sentence_pairs.append(qp_merge_inputs)
                sentence_pairs_idxs.append(idx)
            else:
                start_id = 0
                while start_id < doc_inputs_length:
                    end_id = start_id + max_doc_inputs_length
                    sub_doc_inputs = {k:v[start_id:end_id] for k,v in doc_inputs.items()}
                    start_id = end_id - overlap_tokens_length_implt if end_id < doc_inputs_length else end_id

                    qp_merge_inputs = _merge_inputs(query_inputs, sub_doc_inputs)
                    sentence_pairs.append(qp_merge_inputs)
                    sentence_pairs_idxs.append(idx)
        
        return sentence_pairs, sentence_pairs_idxs


CrossEncoderRanker = CorssEncoderRanker
