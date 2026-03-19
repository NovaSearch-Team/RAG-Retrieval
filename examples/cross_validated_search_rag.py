"""
Cross-Validated Search Integration for RAG Systems

This example demonstrates how to use cross-validated-search to enhance
RAG systems with hallucination-free web search capabilities.

Installation:
    pip install cross-validated-search

Usage:
    python cross_validated_search_rag.py
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Cross-validated search result."""
    title: str
    url: str
    snippet: str
    source: str
    confidence: str


class CrossValidatedRAGRetriever:
    """
    RAG Retriever enhanced with cross-validated web search.
    
    This retriever combines traditional RAG with cross-validated web search
    to provide hallucination-free responses.
    
    Features:
    - Cross-validates facts across multiple search engines
    - Assigns confidence scores (Verified/Likely True/Uncertain/Likely False)
    - Works with any RAG pipeline
    - No API key required
    """
    
    def __init__(
        self,
        use_local_rag: bool = True,
        min_confidence: str = "likely_true",
        max_web_results: int = 5,
    ):
        """
        Initialize the cross-validated RAG retriever.
        
        Args:
            use_local_rag: Whether to use local RAG in addition to web search
            min_confidence: Minimum confidence level (verified, likely_true, uncertain, likely_false)
            max_web_results: Maximum number of web results to return
        """
        self.use_local_rag = use_local_rag
        self.min_confidence = min_confidence
        self.max_web_results = max_web_results
        
        # Import cross-validated-search
        try:
            from cross_validated_search import CrossValidatedSearcher
            self.web_searcher = CrossValidatedSearcher()
        except ImportError:
            raise ImportError(
                "cross-validated-search is required. "
                "Install it with: pip install cross-validated-search"
            )
    
    def retrieve(
        self,
        query: str,
        search_type: str = "text",
    ) -> List[SearchResult]:
        """
        Retrieve cross-validated results for a query.
        
        Args:
            query: The search query
            search_type: Type of search (text, news, images)
        
        Returns:
            List of SearchResult objects with confidence scores
        """
        # Perform cross-validated web search
        results = self.web_searcher.search(
            query=query,
            search_type=search_type,
            max_results=self.max_web_results,
        )
        
        # Convert to SearchResult format
        search_results = []
        for result in results.sources:
            # Filter by minimum confidence
            confidence_map = {
                "verified": 4,
                "likely_true": 3,
                "uncertain": 2,
                "likely_false": 1,
            }
            min_level = confidence_map.get(self.min_confidence, 3)
            result_level = confidence_map.get(results.confidence, 2)
            
            if result_level >= min_level:
                search_results.append(SearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    source=result.engine,
                    confidence=results.confidence,
                ))
        
        return search_results
    
    def retrieve_for_rag(
        self,
        query: str,
        local_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve results for RAG pipeline.
        
        Combines local RAG documents with cross-validated web search.
        
        Args:
            query: The search query
            local_documents: Optional list of local RAG documents
        
        Returns:
            Dictionary with combined results and confidence scores
        """
        web_results = self.retrieve(query)
        
        # Combine with local documents if provided
        all_context = []
        
        # Add web results
        for result in web_results:
            all_context.append({
                "content": result.snippet,
                "source": result.url,
                "title": result.title,
                "confidence": result.confidence,
                "type": "web",
            })
        
        # Add local documents if available
        if self.use_local_rag and local_documents:
            for doc in local_documents:
                all_context.append({
                    "content": doc.get("content", ""),
                    "source": doc.get("source", "local"),
                    "title": doc.get("title", ""),
                    "confidence": "verified",  # Local docs are trusted
                    "type": "local",
                })
        
        return {
            "query": query,
            "context": all_context,
            "confidence": results.confidence if web_results else "uncertain",
            "sources": [r.url for r in web_results],
        }


def example_basic_search():
    """Example: Basic cross-validated search."""
    print("=== Basic Cross-Validated Search ===\n")
    
    retriever = CrossValidatedRAGRetriever()
    results = retriever.retrieve("What is RAG retrieval?")
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result.confidence}] {result.title}")
        print(f"   Source: {result.source}")
        print(f"   URL: {result.url}")
        print(f"   Snippet: {result.snippet[:100]}...")


def example_rag_integration():
    """Example: Integration with RAG pipeline."""
    print("\n=== RAG Integration Example ===\n")
    
    # Local documents (from your RAG database)
    local_docs = [
        {
            "content": "RAG stands for Retrieval-Augmented Generation.",
            "source": "local_knowledge_base",
            "title": "RAG Overview",
        }
    ]
    
    retriever = CrossValidatedRAGRetriever(use_local_rag=True)
    result = retriever.retrieve_for_rag(
        query="What are the latest advances in RAG?",
        local_documents=local_docs,
    )
    
    print(f"Query: {result['query']}")
    print(f"Overall Confidence: {result['confidence']}")
    print(f"\nContext ({len(result['context'])} items):")
    
    for i, ctx in enumerate(result['context'], 1):
        print(f"\n{i}. [{ctx['type']}] {ctx['title']}")
        print(f"   Confidence: {ctx['confidence']}")
        print(f"   Content: {ctx['content'][:100]}...")


def example_confidence_filtering():
    """Example: Filter results by confidence."""
    print("\n=== Confidence Filtering Example ===\n")
    
    # Only accept verified results
    retriever = CrossValidatedRAGRetriever(
        min_confidence="verified",  # Only verified facts
        max_web_results=10,
    )
    
    results = retriever.retrieve("Python 3.12 new features")
    
    verified_count = sum(1 for r in results if r.confidence == "verified")
    print(f"Verified results: {verified_count}/{len(results)}")
    
    for result in results:
        print(f"[{result.confidence}] {result.title}")


if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Validated Search + RAG Integration Examples")
    print("=" * 60)
    
    example_basic_search()
    example_rag_integration()
    example_confidence_filtering()
    
    print("\n" + "=" * 60)
    print("For more information:")
    print("https://github.com/wd041216-bit/cross-validated-search")
    print("=" * 60)