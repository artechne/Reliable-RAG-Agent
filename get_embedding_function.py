from langchain_openai import OpenAIEmbeddings

#TODO: Change the model to huggingface SOTA model
#https://huggingface.co/BM-K/KoSimCSE-roberta-multitask

OPEN_AI_TEXT_EMBDEDING_MODEL_3_LARGE = "text-embedding-3-large"

def get_embedding_function():
    embeddings = OpenAIEmbeddings(model=OPEN_AI_TEXT_EMBDEDING_MODEL_3_LARGE)
    return embeddings