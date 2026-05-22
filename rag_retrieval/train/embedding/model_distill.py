import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from transformers import AutoTokenizer
import torch.nn.functional as F
import subprocess


class DistillEmbedding(nn.Module):
    def __init__(
        self,
        sentence_model=None,
        tokenizer=None,
        mrl_dims=[],
    ):
        super().__init__()

        self.model = sentence_model
        self.tokenizer = tokenizer
        self.mrl_dims = mrl_dims

    def get_embedding(self, input_ids, attention_mask):

        batch_text = {'input_ids': input_ids, 'attention_mask': attention_mask}
        embedding = self.model(batch_text)['sentence_embedding']

        return embedding

    def forward(
        self,
        query_input_ids,  # [batch_size,seq_len]
        query_attention_mask,  # [batch_size,seq_len]
        teacher_embeddings=None,  # [batch_size]
        accelerator=None,
    ):
        student_embeddings = self.get_embedding(query_input_ids, query_attention_mask)

        res_dict = {}
        res_dict['query_embeddings'] = student_embeddings

        if teacher_embeddings is not None:
            #get cosine_loss
            teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)            
            assert student_embeddings.size(0) == teacher_embeddings.size(0)
            cosine_loss = self.cosine_embedding_loss(student_embeddings, teacher_embeddings)
            
            #get similarity_loss and triplet_loss(using mrl)
            teacher_similarity = teacher_embeddings @ teacher_embeddings.transpose(-1, -2)
            triplet_label = torch.where(self.get_score_diff(teacher_embeddings) < 0, 1, -1)
            
            similarity_loss = torch.tensor(0.0, device=student_embeddings.device)
            triplet_loss = torch.tensor(0.0, device=student_embeddings.device)
            for num_dim in self.mrl_dims:
                student_embedding = student_embeddings[..., :num_dim]
                student_embedding = F.normalize(student_embedding, p=2, dim=-1)
                similarity_loss += self.pair_inbatch_similarity_loss(student_embedding, teacher_similarity)
                triplet_loss += self.pair_inbatch_triplet_loss(student_embedding, triplet_label)

            similarity_loss = similarity_loss / len(self.mrl_dims)
            triplet_loss = triplet_loss / len(self.mrl_dims)
            res_dict['loss'] = cosine_loss*10 + similarity_loss*200 + triplet_loss*20
            res_dict['cosine_loss'] = cosine_loss*10 
            res_dict['similarity_loss'] = similarity_loss*200
            res_dict['triplet_loss'] = triplet_loss*20
        return res_dict

    def cosine_embedding_loss(
        self,
        student_embeddings, # [batch_size,dim]
        teacher_embeddings, # [batch_size,dim]
    ):

        # normalization
        student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
        # get cosine loss
        target = torch.ones(student_embeddings.size(0), device=student_embeddings.device)
        loss = F.cosine_embedding_loss(student_embeddings, teacher_embeddings, target)
        return loss

    def pair_inbatch_similarity_loss(
        self,
        student_embeddings, # [batch_size,dim]
        teacher_similarity, # [batch_size,dim]
    ):

        # get mse loss
        #[batch_size,batch_size]<- [batch_size,dim],[dim,batch_size]
        student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)

        loss = F.mse_loss(student_similarity, teacher_similarity)
        return loss

    def pair_inbatch_triplet_loss(
        self,
        student_embeddings, # [batch_size,dim]
        triplet_label, # [batch_size,dim]
        triplet_margin=0.015,
    ):
        # get triplets loss
        loss = F.relu(self.get_score_diff(student_embeddings) * triplet_label + triplet_margin).mean()

        return loss

    def get_score_diff(
        self,
        embedding
    ):
        scores = torch.matmul(embedding, embedding.T)
        scores = scores[torch.triu(torch.ones_like(scores), diagonal=1).bool()]
        score_diff = scores.reshape((1, -1)) - scores.reshape((-1, 1))
        score_diff = score_diff[torch.triu(torch.ones_like(score_diff), diagonal=1).bool()]
        return score_diff


    def encode(
        self,
        sentences,
        device='cpu',
        max_len=512,
        batch_size=512,
        prompt="",
        convert_to_tensor=True,
    ):
        # self.device = device
        # self.to(self.device)
        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [prompt + sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]

            input_ids, attention_mask = self.preprocess(sentences_batch, max_len)

            embeddings = self.forward(input_ids, attention_mask)
            embeddings = embeddings['query_embeddings'].detach().cpu()
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if len(all_embeddings):
            all_embeddings = torch.stack(all_embeddings)
        else:
            all_embeddings = torch.Tensor()

        return all_embeddings

    def preprocess(
        self,
        sentences,
        max_len=512
    ):

        tokens = self.tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)

        input_ids = tokens["input_ids"].to(self.model.device)
        attention_mask = tokens['attention_mask'].to(self.model.device)

        return input_ids, attention_mask

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def save_pretrained(
        self,
        save_dir,
        safe_serialization = False
    ):

        self.model.save(save_dir, safe_serialization=safe_serialization)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        teacher_embedding_dim=None,
        mrl_dims=[],
    ):
        sentence_model = SentenceTransformer(model_name_or_path,trust_remote_code=True)

        # If the embedding model last layer is Normalize, remove it and add a denser layer.
        if isinstance(sentence_model._last_module(), models.Normalize):
            idx = len(sentence_model._modules.keys())            
            sentence_model._modules.pop(str(idx-1))
            print("the embedding model last layer is Normalize, remove it")
        print('init scaling_layer weight.')
        idx = len(sentence_model._modules.keys())
        in_features = sentence_model.get_sentence_embedding_dimension()
        scaling_layer = models.Dense(in_features, teacher_embedding_dim, bias=True, activation_function=torch.nn.modules.linear.Identity())
        sentence_model._modules[str(idx)] = scaling_layer
        print(sentence_model)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        embedding = cls(sentence_model, tokenizer, mrl_dims)
        return embedding


def test_model_embedding():
    cuda_device = 'cuda:7'
    ckpt_path = 'BAAI/bge-base-zh-v1.5'
    embedding = DistillEmbedding.from_pretrained(ckpt_path)
    embedding.to(cuda_device)
    
    input_lst = ['我喜欢中国']

    embedding = embedding.encode(input_lst, device=cuda_device)

    print(len(embedding.tolist()))
    print(embedding.tolist()[0])
    print(len(embedding.tolist()[0]))


if __name__ == "__main__":
    test_model_embedding()
