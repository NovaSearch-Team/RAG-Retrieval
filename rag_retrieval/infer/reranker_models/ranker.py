from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

class BaseRanker(ABC):
    @abstractmethod
    def __init__(self, 
        model_name_or_path: str, 
        verbose: int
    ):
        pass
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: Union[List[str], str] = None,
        batch_size: int = 256,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        **kwargs,
    ):
        """
        Rerank documents with respect to a query.
        
        Args:
            query: Query string.
            docs: A single document string or a list of document strings.
            batch_size: Batch size for model inference.
            top_k: Optional number of top ranked results to return.
            score_threshold: Optional minimum score threshold to filter results.
        """
        pass

    @abstractmethod
    def compute_score(
        self,
        sentences_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        **kwargs,
    ):
        """
        Compute scores for sentence pairs.
        """
        pass
