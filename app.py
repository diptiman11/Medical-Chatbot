from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LC_Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec, PodSpec  
import time  
from langchain_pinecone import PineconeVectorStore  

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()

# Initialize Pinecone using the new API
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Connect to the existing index
index = pc.Index(index_name)

# Create a LangChain Pinecone vector store instance
# docsearch = LC_Pinecone(index, embeddings)

text_field = "text"  
docsearch = PineconeVectorStore(  
    index, embeddings, text_field  
)  

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
