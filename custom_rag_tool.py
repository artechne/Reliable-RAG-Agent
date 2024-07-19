from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from populate_db import get_chroma_db
from get_embedding_function import get_embedding_function
from crewai_tools import BaseTool

QUERY_PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context with markdown format: {question}
"""

RAG_TOP_K = 5

def query_rag(query_text: str):
    # Prepare the DB
    db = get_chroma_db()

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=RAG_TOP_K)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.,
    )

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

class CustomChromaDBRagTool(BaseTool):
    name: str = "Custom Chroma DB Rag Tool"
    description: str = "Retrieve the information about the query in the Chroma DB."

    def _run(self, query: str) -> str:
        # Implementation goes here
        # Prepare the DB
        db = get_chroma_db()

        # Search the DB
        results = db.similarity_search_with_score(query, k=RAG_TOP_K)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        return context_text
        
        #return query_rag(query)