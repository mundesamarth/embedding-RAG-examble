from pinecone import Pinecone
from pinecone import ServerlessSpec
import os
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

## setting up PineCone
pineAPI = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = pineAPI
# Set API keys
# Initialize Pinecone client
pc = Pinecone()

# Define index name
index_name = "alice-in-wonderland"

# Create index if it does not exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 has 384 dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status["ready"]:
      time.sleep(1)

# Get Pinecone Index instance (REQUIRED for LangChain)
index = pc.Index(index_name)

# Loading Embedding

# Load SentenceTransformer embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# Sample book excerpts
book_paragraphs = [
    "Alice was beginning to get very tired of sitting by her sister on the bank.",
    "The White Rabbit was wearing a waistcoat-pocket and took a watch out of it.",
    "Alice followed the Rabbit down a large rabbit-hole and found herself in a strange land.",
    "The Cheshire Cat grinned and said: 'We’re all mad here. I’m mad. You’re mad.'",
    "Alice attended a tea party with the Mad Hatter, the March Hare, and the Dormouse."
]

# Convert paragraphs into LangChain Document objects
documents = [Document(page_content=para) for para in book_paragraphs]

vectorstore = PineconeVectorStore(
    embedding=embedding_model,
    index=index
)

#Add documents to pinecone:
texts = [doc.page_content for doc in documents]  # Extract text from Document objects

# Insert into Pinecone
vectorstore.add_texts(texts=texts)


api_key = os.getenv("LANGCHAIN_API_KEY")
base_url = os.getenv("BASE_URL")

# Load LLM
llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model="meta-llama/llama-3.1-70b-instruct:free",
)

# Set up RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

## Asking Question
# Example queries
queries = [
    "What did the Cheshire Cat say?",
    "Where did Alice go after following the rabbit?"
]

# Answer queries
for query in queries:
    answer = qa_chain.run(query)
    print(f"Query: {query}\nAnswer: {answer}\n")