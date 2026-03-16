import requests

GITHUB_API = "https://api.github.com/search/repositories"


def search_github_repos(query, max_results=10):

    params = {
        "q": f"{query} in:name,description",
        "sort": "stars",
        "order": "desc",
        "per_page": max_results
    }

    response = requests.get(GITHUB_API, params=params)

    data = response.json()

    repos = []

    for repo in data["items"]:

        repos.append({
            "name": repo["name"],
            "description": repo["description"],
            "stars": repo["stargazers_count"],
            "language": repo["language"],
            "url": repo["html_url"]
        })

    return repos