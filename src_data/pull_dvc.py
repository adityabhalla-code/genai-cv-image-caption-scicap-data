import subprocess
from git import Repo
import argparse
import os


def find_repo_root(current_path):
    while not os.path.isdir(os.path.join(current_path, ".git")):
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("Repository root not found.")
        current_path = parent_path
    return current_path


def pull_latest_changes(repo_path):
    try:
        repo = Repo(repo_path)
        if repo.is_dirty(untracked_files=True):
            print("Repository has uncommitted changes.")
        else:
            origin = repo.remotes.origin
            origin.pull()
            print("Pulled the latest changes.")
    except Exception as e:
        print(f"An error occurred while pulling the latest changes: {e}")
        

def pull_data_from_dvc(target):
    try:
        result = subprocess.run(['dvc', 'pull',target], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        print("Data pulled successfully from DVC.")
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode('utf-8'))
        print("An error occurred while pulling data from DVC.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull latest data from DVC")
    parser.add_argument('--target', type=str, required=True, help='The specific folder to pull from DVC')
    args = parser.parse_args()
    
    
    # current_script_path =os.path.dirname(os.path.abspath(__file__))
    # print(f"current_script_path:{current_script_path}")
    # repo_path = find_repo_root(current_script_path)
    # print(f"git repo_path:{repo_path}")
    # pull_latest_changes(repo_path)
    pull_data_from_dvc(args.target)
