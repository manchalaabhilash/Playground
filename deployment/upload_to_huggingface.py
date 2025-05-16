import os
from huggingface_hub import login, HfApi, create_repo
from getpass import getpass

# Configuration
SPACE_NAME = "superkart-sales-predictor"  # Change this to your desired space name
USERNAME = input("Enter your Hugging Face username: ")
REPO_ID = f"{USERNAME}/{SPACE_NAME}"
REPO_TYPE = "space"
SPACE_SDK = "docker"

# Files to upload
FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "superkart_sales_prediction_model_v1_0.joblib",
    "test_model.py"
]

# Login to Hugging Face
print("Please enter your Hugging Face token (create one at https://huggingface.co/settings/tokens):")
token = getpass()
login(token=token)

# Initialize the API
api = HfApi()

# Create the repository if it doesn't exist
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        space_sdk=SPACE_SDK,
        private=False
    )
    print(f"Created new repository: {REPO_ID}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Repository {REPO_ID} already exists. Continuing with upload.")
    else:
        print(f"Error creating repository: {e}")
        exit(1)

# Upload files
for file in FILES_TO_UPLOAD:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE
        )
        print(f"Successfully uploaded {file}")
    else:
        print(f"Warning: File {file} not found, skipping")

print(f"\nDeployment complete! Your API should be available soon at:")
print(f"https://huggingface.co/spaces/{REPO_ID}")