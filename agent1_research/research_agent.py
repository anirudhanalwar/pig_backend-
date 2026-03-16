import json
import os
from datetime import datetime
import ollama

from arxiv_rag_tool import arxiv_research_tool
from github_repo_tool import search_github_repos

MODEL = "llama3:8b"


def build_prompt(query, papers, repos):

    paper_context = ""

    for p in papers:
        paper_context += f"""
Title: {p['paper_title']}
Summary: {p['content']}
"""

    repo_context = ""

    for r in repos:
        repo_context += f"""
Repo: {r['name']}
Description: {r['description']}
Stars: {r['stars']}
"""

    prompt = f"""
You are an AI research analysis system.

Your task is to convert research knowledge into structured JSON.

Return EXACTLY this schema:

{{
  "query": "{query}",
  "timestamp": "",
  "papers": [],
  "github_repositories": [],
  "aggregated_topics": [],
  "methods": [],
  "datasets": [],
  "tools": []
}}

Research Papers:
{paper_context}

GitHub Repositories:
{repo_context}

Rules:
- Only output JSON
- No explanation text
- Do not change schema
"""

    return prompt


def run_llm(prompt):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    text = response["message"]["content"]

    start = text.find("{")
    end = text.rfind("}") + 1

    return text[start:end]


def save_json(data):

    path = os.path.join(os.path.dirname(__file__), "research_output.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("Saved:", path)


def research_agent(query):

    print("Collecting research papers...")

    papers = arxiv_research_tool(query)

    print("Searching GitHub repositories...")

    repos = search_github_repos(query)

    print("Running LLM analysis...")

    prompt = build_prompt(query, papers, repos)

    output = run_llm(prompt)

    try:
        structured = json.loads(output)
    except:
        print("Invalid JSON returned")
        print(output)
        return

    structured["query"] = query
    structured["timestamp"] = str(datetime.now().date())

    save_json(structured)

    return structured


if __name__ == "__main__":

    query = input("Enter research query: ")

    research_agent(query)