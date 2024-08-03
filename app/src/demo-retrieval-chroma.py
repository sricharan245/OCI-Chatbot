from langchain.chains import RetrievalQA
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

# In this demo we will explore using streamlit to input a question to llm and display the response

# Step 1 - setup OCI Gen AI llm
llm = OCIGenAI(
    model_id = "cohere.command-light",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
    model_kwargs = {"max_tokens": 100}
)

# Step 2 - here we connect to a chromadb server we need to run the chroma db server before we connect to it

client = chromadb.HttpClient(host="127.0.0.1")

# Step 3 - here we create embeddings using "cohere.embed-english-v2.0" model
embeddings = OCIGenAIEmbeddings(
    model_id = "cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
)

# Step 4 - here we create a retriever that gets relevant documents (similar in meaning to a query)

db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
retv = db.as_retriever(search_type="similarity", seach_kwargs = {"k": 5})

# Step 5 - here we can explore how we similar documents to the query are returned by printing the document metadata. this step is optional

docs = retv.get_relevant_documents("What is Oracle Cloud Infrastructure (OCI)?")


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document  {i+1}: \n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(docs)

for doc in docs:
    print(doc.metadata)

# Step 6 - here we create a retrieval chain that takes llm, retriever objects and invoke it to get a response to our query

chain = RetrievalQA.from_chain_type(llm=llm, retriever = retv, return_source_documents=True)

response = chain.invoke("What is Oracle Cloud Infrastructure (OCI)?")

print (response)
