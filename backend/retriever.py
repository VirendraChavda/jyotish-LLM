from langchain_community.vectorstores import FAISS
from prompts_and_llm import retriever_embedding_function

retriever_embedder = retriever_embedding_function()

vector_store = FAISS.load_local(
    folder_path=r"D:/Interviews/LangChain/Jyotish/FaissDB",
    embeddings=retriever_embedder,
    allow_dangerous_deserialization=True
)

def retriver_function():
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 40, "lambda_mult": 0.75}
    )

    return retriever

if __name__ == "__main__":
    retriever = retriver_function()
    docs = retriever.get_relevant_documents("Who is the lord of house number 1?")
    for i, d in enumerate(docs, 1):
        print(i, d.metadata.get("doc_id") or d.metadata.get("source"), d.page_content[:120].replace("\n"," "))