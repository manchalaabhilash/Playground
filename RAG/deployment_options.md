# Deployment Options for RAG System

## Hugging Face Spaces (Recommended)

Hugging Face Spaces offers free hosting for ML applications with Docker support.

### Steps to deploy:

1. Create a Hugging Face account at https://huggingface.co/
2. Create a new Space:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose a name (e.g., "ml-textbook-rag")
   - Select "Docker" as the SDK
   - Set visibility to "Public" or "Private"

3. Prepare your repository:
   - Create a `README.md` file with Space metadata at the top:
     ```
     ---
     title: ML Textbook RAG
     emoji: 📚
     colorFrom: blue
     colorTo: indigo
     sdk: docker
     app_port: 8501
     pinned: false
     ---
     ```
   - Modify your `docker-compose.yml` to use a single service

4. Push your code:
   ```bash
   git add .
   git commit -m "Initial commit"
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ml-textbook-rag
   git push space main
   ```

## Render

Render offers a free tier for web services with Docker support.

### Steps to deploy:

1. Create a Render account at https://render.com/
2. Create a new Web Service:
   - Connect your GitHub repository
   - Select "Docker" as the environment
   - Choose a name and region
   - Set the instance type to "Free"

3. Configure environment variables in the Render dashboard

## Google Cloud Run

Google Cloud Run offers a free tier (with credit limits).

### Steps to deploy:

1. Create a Google Cloud account and set up billing
2. Install Google Cloud SDK
3. Build and push your Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/rag-app
   ```
4. Deploy to Cloud Run:
   ```bash
   gcloud run deploy rag-app --image gcr.io/YOUR_PROJECT_ID/rag-app --platform managed
   ```

## Important Considerations

For any cloud deployment:

1. **Ollama Integration**: You'll need to either:
   - Include Ollama in your Docker setup
   - Use a cloud-based LLM API instead (like OpenAI or Hugging Face Inference API)

2. **Persistence**: Configure persistent storage for:
   - Uploaded PDFs
   - Vector database

3. **Resource Limits**: Free tiers have memory/CPU limitations that may affect performance