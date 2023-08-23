import os
import subprocess
import shutil


def git_clone(repo_url, destination_folder=None):
    if destination_folder:
        command = ["git", "clone", repo_url, destination_folder]
    else:
        command = ["git", "clone", repo_url]

    try:
        subprocess.run(command, check=True)
        print("Repository cloned successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def move_contents(src_folder, dest_folder):
    # List all the files and subfolders in the source folder
    for item in os.listdir(src_folder):
        # Construct full file or folder path
        src_item_path = os.path.join(src_folder, item)

        # Destination path
        dest_item_path = os.path.join(dest_folder, item)

        # Move the item to the destination folder
        shutil.move(src_item_path, dest_item_path)


# Example usage
langchain_url = "https://github.com/langchain-ai/langchain.git"
try:
    git_clone(langchain_url)
except Exception:
    pass

twitter_url = "https://github.com/twitter/the-algorithm"
try:
    git_clone(twitter_url)
except Exception:
    pass

# Example Usage:
src = "langchain"
dest = "./"
move_contents(src, dest)
