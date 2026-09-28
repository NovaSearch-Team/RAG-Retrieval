from typing import Union, Optional, List, Iterator, Dict, Any
from pydantic import BaseModel

try:
    from pydantic import model_validator
    _PYDANTIC_V2 = True
except ImportError:
    from pydantic import validator
    _PYDANTIC_V2 = False


class Result(BaseModel):
    doc_id: Union[int, str]
    text: str
    score: Optional[float] = None
    rank: Optional[int] = None

    if _PYDANTIC_V2:
        @model_validator(mode="after")
        def check_score_or_rank_exists(self):
            if self.rank is None and self.score is None:
                raise ValueError("Either score or rank must be provided.")
            return self
    else:
        @validator("rank", always=True)
        def check_score_or_rank_exists(cls, v, values):
            if v is None and values.get("score") is None:
                raise ValueError("Either score or rank must be provided.")
            return v

    def to_dict(self) -> Dict[str, Any]:
        """Returns the result as a dictionary."""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class RankedResults(BaseModel):
    results: List[Result]
    query: str
    has_scores: bool = False

    def __len__(self) -> int:
        """Returns the total number of ranked results."""
        return len(self.results)

    def __iter__(self) -> Iterator[Result]:
        """Iterates over ranked results."""
        return iter(self.results)

    def __getitem__(self, index: Union[int, slice]) -> Union[Result, "RankedResults"]:
        """Supports indexing (returning Result) and slicing (returning RankedResults)."""
        if isinstance(index, slice):
            return RankedResults(
                results=self.results[index],
                query=self.query,
                has_scores=self.has_scores,
            )
        return self.results[index]

    def results_count(self) -> int:
        """Returns the total number of results."""
        return len(self.results)

    def top_k(self, k: int) -> List[Result]:
        """Returns the top k results based on the score, if available, or rank."""
        if self.has_scores:
            return sorted(
                self.results,
                key=lambda x: x.score if x.score is not None else float("-inf"),
                reverse=True,
            )[:k]
        else:
            return sorted(
                self.results,
                key=lambda x: x.rank if x.rank is not None else float("inf"),
            )[:k]
    
    def get_score_by_docid(self, doc_id: Union[int, str]) -> Optional[float]:
        """Fetches the score of a result by its doc_id using a more efficient approach."""
        result = next((r for r in self.results if r.doc_id == doc_id), None)
        return result.score if result else None

    def filter_by_score(self, min_score: float) -> "RankedResults":
        """Returns a new RankedResults with only results having score >= min_score."""
        filtered = [
            r for r in self.results
            if r.score is not None and r.score >= min_score
        ]
        return RankedResults(
            results=filtered,
            query=self.query,
            has_scores=self.has_scores,
        )

    def filter_by_rank(self, max_rank: int) -> "RankedResults":
        """Returns a new RankedResults with only results having rank <= max_rank."""
        filtered = [
            r for r in self.results
            if r.rank is not None and r.rank <= max_rank
        ]
        return RankedResults(
            results=filtered,
            query=self.query,
            has_scores=self.has_scores,
        )

    def get_texts(self) -> List[str]:
        """Returns list of document text contents in current ranked order."""
        return [r.text for r in self.results]

    def get_doc_ids(self) -> List[Union[int, str]]:
        """Returns list of document IDs in current ranked order."""
        return [r.doc_id for r in self.results]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the RankedResults object to a dictionary."""
        return {
            "query": self.query,
            "has_scores": self.has_scores,
            "results": [r.to_dict() for r in self.results],
        }

    def to_list(self) -> List[Dict[str, Any]]:
        """Returns the ranked results as a list of dictionaries."""
        return [r.to_dict() for r in self.results]
