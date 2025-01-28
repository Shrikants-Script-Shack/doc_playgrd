import os
import re
import time
import faiss
import streamlit as st
import mlflow
from dotenv import load_dotenv  # Import dotenv to load environment variables
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq  
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Set up MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("TransInnovate")

# Streamlit UI
st.title("TransInnovate :: MyAgent")
st.markdown("TestV6")

LOG_FILE_PATH = "IMGBRANOLTP.txt"

# Function to load log file
def load_latest_logs():
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "Log file not found."

# Function to clean logs
def clean_logs(log_data):
    log_lines = log_data.split("\n")
    cleaned_lines = []
    
    for line in log_lines:
        line = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', line)  # Remove timestamps
        line = re.sub(r'[^\w\s:]', '', line).strip()  # Remove special characters
        if len(line) > 5:
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

cleaned_logs = clean_logs(load_latest_logs())

# Split logs into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(cleaned_logs)

# Use Hugging Face Inference API for embeddings
embedding_model = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/msmarco-MiniLM-L-6-v3",
    api_key=HUGGINGFACE_API_KEY
)

# Store embeddings in FAISS
vector_store = FAISS.from_texts(chunks, embedding_model)
faiss.write_index(vector_store.index, "faiss_index.bin")

retriever = vector_store.as_retriever()

# Load error keywords
error_keywords_file = "error_keywords.txt"
with open(error_keywords_file, "r") as f:
    error_keywords = [line.strip() for line in f.readlines() if line.strip() and not line.endswith(":")]

# Initialize Groq LLM
groq_llm = ChatGroq(model_name="mixtral-8x7b-32768", api_key=GROQ_API_KEY)

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a log analysis assistant. Given the following logs:
    {context}
    
    Answer the user's question in **4-5 concise lines**.
    If there is any error code mentioned, please include it in the first line.
    Additionally, provide the **last 4-5 Java files from the stack trace**, if available.
    
    User's Question:
    {question}
    """
)
def retrieve_logs_with_error_context(question):
    error_keywords_str = ", ".join(error_keywords)
    modified_query = f"{question}, these errors may contain keywords: {error_keywords_str}"
    retrieved_docs = retriever.invoke(modified_query)
    retrieved_context = " ".join([doc.page_content for doc in retrieved_docs])
    return retrieved_context
# Query the logs using Groq
def ask_logs(question):
    start_time = time.time()
    with mlflow.start_run():
        retrieved_context = retrieve_logs_with_error_context(question)
        formatted_prompt = prompt_template.format(context=retrieved_context, question=question)
        
        # Query Groq
        response_start = time.time()
        response = groq_llm.invoke(formatted_prompt)
        response_time = time.time() - response_start
        total_time = time.time() - start_time
        
        # Log in MLflow
        mlflow.log_param("query", question)
        mlflow.log_param("retrieved_context", str(retrieved_context))
        mlflow.log_param("response", str(response))
        mlflow.log_metric("retrieval_time", response_time)
        mlflow.log_metric("total_processing_time", total_time)
        
        mlflow.end_run()
    
    return response

# Streamlit UI for querying logs
user_query = st.text_input("Enter your query:", placeholder="What is the error in the logs?")
if st.button("Ask"):
    if user_query:
        bot_response = ask_logs(user_query)
        st.write("**1stVectorDB:**", bot_response)
