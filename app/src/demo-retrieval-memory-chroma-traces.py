from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

import os
from uuid import uuid4

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test111 - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<input-api-key>" # TODO: Input your api key

# In this demo we will explore using RetrievalQA chain to retrieve releavant documents and sent these as a context in a query
# We will use Chroma vectorstore

#Step 1 - set up OCI Gen AI llm

# use default authN method API key
llm = OCIGenAI(
    model_id = "cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
    model_kwargs = {"max_tokens": 400}
)

# Step 2: here we connect to a chromadb server. we need to run the chromadb server before we connect to it

client = chromadb.HttpClient(host="127.0.0.1", settings= Settings(allow_reset = True))

# Step 3 - here we create embeddings using "cohere.embed-english-v2.0" model
embeddings = OCIGenAIEmbeddings(
    model_id = "cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
)

# Step 4 - here we create a retriever that gets relevant documents (similar in meaning to a query)

# db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity", seach_kwargs = {"k": 8})

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document  {i+1}: \n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# Step 5 - here we create a memory to remember chat messages

memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, output_key='answer')

# Step 6 - here we create a chain that uses llm, retriever and memory

qa = ConversationalRetrievalChain.from_llm(llm, retriever = retv, memory= memory, return_source_documents = True)

response = qa.invoke({"question": "Tell us about Oracle Cloud Infrastructure."})
print(memory.chat_memory.messages)

response = qa.invoke({"question": "Which module tells about databases"})
print(memory.chat_memory.messages)

print(response)