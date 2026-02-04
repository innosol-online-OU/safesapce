import os
import requests
import argparse
import sys

def report_status(token, owner, repo, sha, state, target_url, context, description):
    url = f"https://api.github.com/repos/{owner}/{repo}/statuses/{sha}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "state": state,
        "target_url": target_url,
        "description": description,
        "context": context
    }
    
    print(f"Posting status to {url}...")
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print(f"✅ Status '{state}' reported successfully for {context}.")
    except Exception as e:
        print(f"❌ Failed to report status: {e}")
        if 'response' in locals():
            print(f"Response: {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Report CI status to GitHub')
    parser.add_argument('--token', required=True, help='GitHub Personal Access Token')
    parser.add_argument('--owner', required=True, help='GitHub Repo Owner')
    parser.add_argument('--repo', required=True, help='GitHub Repo Name')
    parser.add_argument('--sha', required=True, help='Commit SHA')
    parser.add_argument('--state', required=True, choices=['pending', 'success', 'failure', 'error'], help='Status state')
    parser.add_argument('--url', required=True, help='Link to Gitea CI Run')
    parser.add_argument('--context', default='Gitea/CI', help='Status context label')
    parser.add_argument('--desc', default='Gitea CI Run', help='Status description')
    
    args = parser.parse_args()
    
    report_status(args.token, args.owner, args.repo, args.sha, args.state, args.url, args.context, args.desc)
