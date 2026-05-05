# Hello Python v2 - Interactive Streamlit App with Hugging Face

This project features an interactive Streamlit web application that uses a lightweight Hugging Face `distilgpt2` model for text generation. Users can enter their name and a message, and the AI will generate a response.

## Project Structure

- `app.py`: The main Streamlit application script that loads the Hugging Face model and defines the UI.
- `Dockerfile`: Defines the Docker image for building and running the Streamlit application.
- `requirements.txt`: Lists Python dependencies, including `streamlit`, `transformers`, and `torch`.
- `.dockerignore`: Specifies files and directories to exclude from the Docker image.

## Getting Started

Follow these steps to set up and run the interactive Streamlit application using Docker.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.

### 1. Ensure Project Files

Make sure you have the following files in your project root directory:

- `app.py`
- `requirements.txt` (containing `streamlit`, `transformers`, `torch`)
- `Dockerfile`
- `.dockerignore`

### 2. Build the Docker Image

Navigate to your project root directory in your terminal and build the Docker image. We'll tag it as `hello-py-v2-interactive`:

```bash
docker build -t hello-py-v2-interactive .
```

### 3. Run the Docker Container

Once the image is built, run the Docker container. Streamlit's default port is 8501, so we'll map this port from the container to your host machine:

```bash
docker run -p 8501:8501 hello-py-v2-interactive
```

### 4. Access the Streamlit Application

After running the container, open your web browser and navigate to:

```
http://localhost:8501
```

You will see the interactive Streamlit application where you can enter your name and a message to get a response from the Hugging Face `distilgpt2` model.
