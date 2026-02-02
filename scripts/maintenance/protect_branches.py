import os
import sys
import base64
import json
import time

# Configuration
GITEA_URL = "http://localhost:3000"
USERNAME = "pipeline_bot"
PASSWORD = "SafeSpace!2026"
REPO_OWNER = "AJamal27891"
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
    session.auth = (USERNAME, PASSWORD)
    
    # Check if we need to create the user/org AJamal27891 first? 
    # Usually Gitea creates user on first login, but since we use API, we might need to create organization/user
    # But for simplicity, let's try to create the repo under 'anti_admin' if AJamal27891 doesn't exist, 
    # OR we can assume the user wants it under the same name.
    
    # 1. Check/Create Repository
    api_url = f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}"
    resp = session.get(api_url)
    
    if resp.status_code == 404:
        print(f"Repo {owner}/{repo_name} not found. Attempting to create...")
        # Try creating under authenticated user first
        create_payload = {
            "name": repo_name,
            "private": False,
            "auto_init": False 
        }
        # If owner is different from anti_admin, we might need to create an Org. 
        # For now, let's create it under anti_admin if we can't create it under owner.
        if owner == USERNAME:
            create_url = f"{GITEA_URL}/api/v1/user/repos"
            resp = session.post(create_url, json=create_payload)
        else:
            # Check if Org exists
            org_url = f"{GITEA_URL}/api/v1/orgs/{owner}"
            if session.get(org_url).status_code == 404:
                 print(f"Organization {owner} not found. creating...")
                 session.post(f"{GITEA_URL}/api/v1/orgs", json={"username": owner, "visibility": "public"})
            
            create_url = f"{GITEA_URL}/api/v1/orgs/{owner}/repos"
            resp = session.post(create_url, json=create_payload)

        if resp.status_code == 201:
            print("Repo created successfully.")
        else:
            print(f"Failed to create repo: {resp.status_code} {resp.text}")
            # Try falling back to user repo
            if owner != USERNAME:
                 print("Fallback: Creating under anti_admin")
                 owner = USERNAME
                 create_url = f"{GITEA_URL}/api/v1/user/repos"
                 resp = session.post(create_url, json=create_payload)
                 if resp.status_code != 201:
                     sys.exit(1)
            else:
                sys.exit(1)

    # 1.5 Enable Actions (Crucial!)
    print("Enabling Actions...")
    session.patch(f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}", json={"has_actions": True})

    # 2. Define Protection Rules - BUT first we need branches.
    # If the repo is empty, we cannot protect branches.
    # We will simply verify we can access it. Pushing code happens from shell.
    
    # Check if branches exist
    branches_url = f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}/branches"
    branches_resp = session.get(branches_url)
    if branches_resp.status_code == 200 and not branches_resp.json():
        print("Repo is empty. Push code first.")
        return

    branches = ["main", "dev"]
    for branch in branches:
        print(f"Protecting branch: {branch}...")
        rule_url = f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}/branch_protections/{branch}"
        
        payload = {
            "branch_name": branch,
            "enable_push": False,
            "enable_push_whitelist": False,
            "push_whitelist_usernames": [],
            "enable_merge_whitelist": True,
            "merge_whitelist_usernames": [USERNAME],
            "enable_status_check": True,
            "status_check_contexts": ["security-scan", "code-quality", "unit-tests"],
            "required_approvals": 0,
            "enable_approvals_whitelist": False,
            "block_on_outdated_branch": False
        }
        
        check = session.get(rule_url)
        if check.status_code == 200:
            print(f"Updating existing rule for {branch}")
            resp = session.patch(rule_url, json=payload)
        else:
            print(f"Creating new rule for {branch}")
            resp = session.post(f"{GITEA_URL}/api/v1/repos/{owner}/{repo_name}/branch_protections", json=payload)
        
        if resp.status_code in [200, 201]:
            print(f"SUCCESS: Protected {branch}")
        else:
            # Fails if branch doesn't exist
            print(f"Warning: Could not protect {branch} (might not exist yet).")

if __name__ == "__main__":
    main()

