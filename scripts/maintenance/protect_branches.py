import os
import sys
import base64
import json
import time

# Configuration
GITEA_URL = "https://gitea.innosol.online"
TOKEN = os.getenv("GITEA_PUSH_TOKEN") or "e76029cf2d499c7fcba0f69e2149ec6d87b88a18"

if not TOKEN:
    print("❌ Error: GITEA_PUSH_TOKEN environment variable is not set.")
    sys.exit(1)
REPO_OWNER = "Madscientist72"
REPO_NAME = "safespace"

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

def get_repo_details():
    # Only rely on hardcoded for local Gitea as git remote points to github
    return REPO_OWNER, REPO_NAME

def main():
    owner, repo_name = get_repo_details()
    print(f"Targeting Repository: {owner}/{repo_name}")

    session = requests.Session()
    session.headers.update({"Authorization": f"token {TOKEN}"})
    
    # Repo Check/Create
    api_url = f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}"
    resp = session.get(api_url)
    
    if resp.status_code == 404:
        print(f"Creating {owner}/{repo_name}...")
        create_payload = {"name": repo_name, "private": False, "auto_init": False}
        
        # Try User then Org
        if owner == USERNAME:
            session.post(f"{GITEA_URL}/api/v1/user/repos", json=create_payload)
        else:
            if session.get(f"{GITEA_URL}/api/v1/orgs/{owner}").status_code == 404:
                 session.post(f"{GITEA_URL}/api/v1/orgs", json={"username": owner, "visibility": "public"})
            session.post(f"{GITEA_URL}/api/v1/orgs/{owner}/repos", json=create_payload)

    # Enable Actions
    session.patch(f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}", json={"has_actions": True})

    # Protect Branches
    branches_resp = session.get(f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}/branches")
    if branches_resp.status_code == 200 and not branches_resp.json():
        print("Repo empty. Push code first.")
        return

    for branch in ["main", "dev"]:
        print(f"Protecting {branch}...")
        rule_url = f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}/branch_protections/{branch}"
        
        payload = {
            "branch_name": branch,
            "enable_push": False,
            "enable_push_whitelist": False,
            "push_whitelist_usernames": [],
            "enable_merge_whitelist": True,
            "merge_whitelist_usernames": [REPO_OWNER],
            "enable_status_check": True,
            "status_check_contexts": ["security-scan", "code-quality", "unit-tests"],
            "required_approvals": 0,
            "enable_approvals_whitelist": False,
            "block_on_outdated_branch": False
        }
        
        if session.get(rule_url).status_code == 200:
            session.patch(rule_url, json=payload)
        else:
            session.post(f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}/branch_protections", json=payload)

if __name__ == "__main__":
    main()

