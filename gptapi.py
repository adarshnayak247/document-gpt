import openai
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"

# URL to extract data
url = "https://brainlox.com/courses/category/technical"

# Create WebBaseLoader for scraping
loader = WebBaseLoader(web_paths=[url])

# Load the documents
docs = loader.load()

# Print the number of documents loaded
print(f"{len(docs)} documents loaded")

# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Generate embeddings and store them in Chroma vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Initialize Flask app
app = Flask(__name__)

# Define your LLM (Language Model) with the provided API key
llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, openai_api_key=openai.api_key)

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Write with simple language.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the chain for processing the question
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# API to handle conversation
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    # Get response from the chain
    result = rag_chain_with_source.invoke(question)
    
    # Format the result to markdown or any preferred format
    response = format_to_markdown(result)
    
    return jsonify({'response': response})

# Helper function to format the output as markdown
def format_to_markdown(data):
    markdown_output = f"Question: {data['question']}\n\nAnswer:\n{data['answer']}\n\nSources:\n\n"
    for i, doc in enumerate(data["context"], start=1):
        page_content = doc.page_content.split("\n")[0]  # Get the first line of content
        source_link = doc.metadata["source"]
        markdown_output += f"[[{i}]({source_link})] {page_content}\n\n"
    return markdown_output

if __name__ == '__main__':
    app.run(debug=True)
