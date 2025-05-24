from flask import Flask, render_template, request
from src.helper import load_pdf_file, text_split, download_ollama_embeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src.prompt import *

app = Flask(__name__)

# Load PDF and split into chunks
pdf_path = "Data\Retrievel-Augmented-Generation-for-NLP.pdf"
pdf_text = load_pdf_file(pdf_path)
chunks = text_split(pdf_text)

# Load Ollama embeddings
embeddings = download_ollama_embeddings()

# Create FAISS vector store
docsearch = FAISS.from_texts(texts=chunks, embedding=embeddings)

# Initialize LLaMA chat model
llm = ChatOllama(model="mistral")

# Setup retriever with multi-query prompt
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setup chat prompt
# prompt = ChatPromptTemplate.from_template(template)

# Create chain: retrieval -> prompt -> LLM -> parse output
# llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
