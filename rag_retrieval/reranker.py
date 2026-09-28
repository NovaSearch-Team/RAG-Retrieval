from typing import Optional
import os

from rag_retrieval.infer.reranker_models import AVAILABLE_RANKERS

os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')


DEFAULTS_MODEL_CLASS_TYPE = {
    'CorssEncoderRanker',
    'CrossEncoderRanker',
    'ColBERTRanker',
    'LLMRanker',
}

DEFAULTS_MODEL_TYPE = {
    'llm',
    'colbert',
    'cross-encoder',
    'cross_encoder',
}

DEPS_MAPPING = {
    'CorssEncoderRanker': 'transformers',
    'CrossEncoderRanker': 'transformers',
    'ColBERTRanker': 'transformers',
    'LLMRanker': 'transformers',
}


def _get_model_type(
    model_name: str, 
    model_type: Optional[str] = None,
):
    if model_type is not None:
        model_type = model_type.lower()
        model_type_to_class = {
            "llm": "LLMRanker",
            "colbert": "ColBERTRanker",
            "cross-encoder": "CrossEncoderRanker",
            "cross_encoder": "CrossEncoderRanker",
            "crossencoderranker": "CrossEncoderRanker",
            "corssencoderranker": "CorssEncoderRanker",
            "llmranker": "LLMRanker",
            "colbertranker": "ColBERTRanker",
        }
        return model_type_to_class.get(model_type, model_type)
    else:
        model_name = model_name.lower()
        model_name_to_class = {
            "bge-reranker-v2-gemma": "LLMRanker",
            "bge-reranker-v2-minicpm-layerwise": "LLMRanker",
            "bge-reranker-base": "CrossEncoderRanker",
            "bge-reranker-large": "CrossEncoderRanker",
            "bge-reranker-v2-m3": "CrossEncoderRanker",
            "bce": "CrossEncoderRanker",
            "bge-m3": "ColBERTRanker",
        }

        for key, value in model_name_to_class.items():
            if key in model_name:
                return value


def Reranker(
    model_name: str,
    model_type: Optional[str] = None,
    verbose: int = 1,
    **kwargs
):
    # Infer the model class of the reranker (by model_name or model_type)
    model_class_type = _get_model_type(model_name, model_type)
    
    if model_class_type not in DEFAULTS_MODEL_CLASS_TYPE:
        if model_type is not None:
            print(f"Model type is not supported, please input one of {str(DEFAULTS_MODEL_TYPE)}")
            return None
        else:
            print(
                "Warning: Model type could not be auto-mapped with the defaults list. Defaulting to cross-encoder."
            )
            print(
                "If your model is NOT intended to be run as a one-label cross-encoder, please reload it and specify the model_type!",
                "Otherwise, you may ignore this warning. You may specify `model_type='cross-encoder'` to suppress this warning in the future.",
            )
            model_class_type = "CrossEncoderRanker"
    try:
        print(f"Loading {model_class_type} model {model_name}")
        return AVAILABLE_RANKERS[model_class_type](model_name, verbose=verbose, **kwargs)
    except KeyError:
        print(
            f"You don't have the necessary dependencies installed to use {model_class_type}."
        )
        return None
