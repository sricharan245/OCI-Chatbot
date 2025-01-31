from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.embeddings import CohereEmbeddings

pdf_loader = PyPDFDirectoryLoader("app/assets/pdf-docs") # I used OCI IT enterprise document downloaded from Oracle website as I did not find pdf docs shown in the course.

loaders = [pdf_loader]
# print(loaders)
documents = []

for loader in loaders:
    # print(loader.load())
    documents.extend(loader.load())

text_spliter = RecursiveCharacterTextSplitter(chunk_size = 2500, chunk_overlap = 100)
all_documents = text_spliter.split_documents(documents)

print(f"Total number of documents: {len(all_documents)}")

# Step 1 - setup OCI Generative AI llm

embeddings = OCIGenAIEmbeddings(
    model_id = "cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
    model_kwargs = {"truncate": True}
)

# Step 2 - since OCIGenAIEmbeddings accepts only 96 documents in one run, we will input documents in batches

# Set the batch size

batch_size = 96

# Calculate the number of batches
num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)

db = Chroma(embedding_function=embeddings, persist_directory="./chromadb")
retv = db.as_retriever()

# texts = ["FAISS is an important library", "Langchain supports FAISS"]
# db = FAISS.from_texts(texts, embeddings)
# retv = db.as_retriever()

# Iterate over batches
for batch_num in range(num_batches):
    print("Started")
    # Calculate start and end indices for the current batch
    start_index = batch_num * batch_size
    end_index = (batch_num + 1) * batch_size
    # Extract documents for the current batches
    batch_documents = all_documents[start_index:end_index]
    # Yout code to process each document goes here
    retv.add_documents(batch_documents)
    print("start and end: ", start_index, end_index)

# Step 4: here we persist the collection
db.persist()
# db.save_local("faiss_index")