from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
#https://github.com/ssisOneTeam/Korean-Embedding-Model-Performance-Benchmark-for-Retriever

OPEN_AI_TEXT_EMBDEDING_MODEL_3_LARGE = "text-embedding-3-large"
HUGGING_FACE_MODEL_NAME = "jhgan/ko-sroberta-multitask"

def get_embedding_function(model_source: str = "huggingface"):
    if model_source == "huggingface":
        return HuggingFaceEmbeddings(model_name=HUGGING_FACE_MODEL_NAME)
    elif model_source == "openai":
        return OpenAIEmbeddings(model=OPEN_AI_TEXT_EMBDEDING_MODEL_3_LARGE)
    else:
        raise ValueError(f"Unknown model source: {model_source}")