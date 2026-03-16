import arxiv
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import uuid

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./vector_store")

collection = client.get_or_create_collection(name="research_papers")


def search_arxiv(topic, max_results=20):

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in search.results():

        papers.append({
            "id": str(uuid.uuid4()),
            "title": result.title,
            "abstract": result.summary,
            "pdf_url": result.pdf_url
        })

    return papers


def build_chunks(papers):

    chunks = []

    for p in papers:

        text = f"""
Title: {p['title']}

Abstract:
{p['abstract']}
"""

        chunks.append({
            "id": p["id"],
            "text": text,
            "metadata": {
                "title": p["title"],
                "pdf": p["pdf_url"]
            }
        })

    return chunks


def store_chunks(chunks):

    for c in tqdm(chunks):

        embedding = embedding_model.encode(c["text"]).tolist()

        collection.add(
            ids=[c["id"]],
            embeddings=[embedding],
            documents=[c["text"]],
            metadatas=[c["metadata"]]
        )


def retrieve_research(query, top_k=5):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):

        retrieved.append({
            "paper_title": meta["title"],
            "content": doc,
            "pdf": meta["pdf"]
        })

    return retrieved


def arxiv_research_tool(query):

    papers = search_arxiv(query)

    chunks = build_chunks(papers)

    store_chunks(chunks)

    return retrieve_research(query)