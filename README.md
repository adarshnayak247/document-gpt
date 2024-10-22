# Chat with URL Content - FastAPI Application

This project is a FastAPI-based web application that allows users to extract content from a URL, process the text, and answer questions based on the extracted content using the `llama3.2` model from Ollama. The application also utilizes FAISS for efficient similarity search and a front-end HTML form for easy interaction.

## Prerequisites

Before starting, make sure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- FastAPI and its dependencies
- Ollama model service running on your machine

## Getting Started

### Step 1: Clone the repository

First, clone this repository to your local machine using the command:

```bash
https://github.com/adarshnayak247/document-gpt.git
```

## Step 2: Create a Virtual Environment
Navigate to the project folder and create a virtual environment:
```bash
cd your-repo-name
python3 -m venv venv
venv\Scripts\activate
```

## Step 3: Install Dependencies
With the virtual environment activated, install the dependencies listed in requirements.txt:

```bash
pip install -r requirements.txt
```
## The requirements.txt contains the necessary libraries including:

- FastAPI
- Langchain
- FAISS
- Ollama LLM
- HTTPX
- BeautifulSoup4
- Uvicorn

## Step 4: Start Ollama Service
Ensure that you have Ollama's model service running locally on http://localhost:11434. You can install and run Ollama with the following steps:
```bash
ollama start
ollama pull llama3.2

```

## Step 5: Running the FastAPI Application
To start the FastAPI server, run the following command:
```bash
uvicorn main:app --reload

```


## Step 6: Access the Front-End
Once the FastAPI application is running, you can access the front-end HTML form by visiting:
``` bash
http://127.0.0.1:8000/
```

## output 
![Screenshot Description](https://github.com/adarshnayak247/document-gpt/blob/main/Screenshot%202024-10-22%20134635.png)
![Screenshot Description](https://github.com/adarshnayak247/document-gpt/blob/main/Screenshot%202024-10-22%20073348.png)

