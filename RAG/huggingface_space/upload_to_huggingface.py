import os
from huggingface_hub import login, HfApi, create_repo
from getpass import getpass

# Configuration
SPACE_NAME = "ml-textbook-rag"  # Change this to your desired space name
USERNAME = input("Enter your Hugging Face username: ")
REPO_ID = f"{USERNAME}/{SPACE_NAME}"
REPO_TYPE = "space"
SPACE_SDK = "docker"

# Files to upload
FILES_TO_UPLOAD = [
    "README.md",
    "requirements.txt",
    "Dockerfile",
    ".env",
    "src",
    "api",
    "ui",
    "data"
]

# Login to Hugging Face
print("Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):")
token = getpass()
login(token=token)

# Initialize API
api = HfApi()

# Create or get repository
try:
    print(f"Creating Space: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        space_sdk=SPACE_SDK,
        private=False
    )
except Exception as e:
    print(f"Note: {e}")
    print("Continuing with upload...")

# Upload files
for file in FILES_TO_UPLOAD:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        if os.path.isdir(file):
            api.upload_folder(
                folder_path=file,
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE
            )
        else:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE
            )
        print(f"Successfully uploaded {file}")
    else:
        print(f"Warning: File {file} not found, skipping")

print(f"\nDeployment complete! Your RAG system should be available soon at:")
print(f"https://huggingface.co/spaces/{REPO_ID}")