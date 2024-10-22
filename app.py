from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from dotenv import load_dotenv
import httpx
from bs4 import BeautifulSoup
import os

load_dotenv()

app = FastAPI()

# Serve static files for HTML front-end
app.mount("/folder", StaticFiles(directory="folder"), name="static")
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

# Function to fetch and extract text from the <main> tag of the HTML
async def fetch_text(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        html_content = response.text
        
        # Parse HTML and extract content inside <main> tag
        soup = BeautifulSoup(html_content, 'html.parser')
        main_tag = soup.find('div')  # Find the main tag
        
        if main_tag is None:
            raise HTTPException(status_code=400, detail="No <main> tag found in the HTML.")
        
        return main_tag.get_text(strip=True)  # Get text inside <main> tag

# Function to get the first 500 words from text
def extract_first_500_words(text):
    words = text.split()[:500]  # Get the first 500 words
    return ' '.join(words)  # Join them back into a single string

# Function to get text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True) 
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = OllamaEmbeddings(model="llama3.2", base_url="http://localhost:11434")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available in the context, say: "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# API endpoint to handle user queries
@app.post("/ask")
async def ask_question(url: str = Form(...), user_question: str = Form(...)):
    try:
        raw_text = await fetch_text(url)
        print(f"Fetched raw text length: {len(raw_text)}")

        if len(raw_text) == 0:
            raise HTTPException(status_code=400, detail="No text fetched from the URL.")

        # Extract the first 500 words from the raw text
        limited_text = extract_first_500_words(raw_text)
        print(f"First 500 words:\n{limited_text}")  # Print the first 500 words

        text_chunks = get_text_chunks(limited_text)

        if not text_chunks:
            raise HTTPException(status_code=400, detail="No valid text chunks found.")

        get_vector_store(text_chunks)

        # Load the vector store and find relevant documents
        embeddings = OllamaEmbeddings(model="llama3.2", base_url="http://localhost:11434")
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        return {"response": response["output_text"]}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred: " + str(e))

# HTML for the front-end
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat with URL Content</title>
</head>
<body>
    <h2>Ask a Question from the URL Content</h2>
    <form action="/ask" method="post">
        <label for="url">Enter URL:</label><br>
        <input type="text" id="url" name="url" required><br><br>
        <label for="question">Your Question:</label><br>
        <input type="text" id="question" name="user_question" required><br><br>
        <input type="submit" value="Ask">
    </form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
