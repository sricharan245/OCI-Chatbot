import os
from uuid import uuid4
import langsmith
from langchain import smith
from langchain.smith import RunEvalConfig

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OCIGenAI

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<api-key>" # TODO: Input your api key



# use default authN method API key
llm = OCIGenAI(
    model_id = "cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
    model_kwargs = {"max_tokens": 400}
)

# here we create embeddings using "cohere.embed-english-v2.0" model
embeddings = OCIGenAIEmbeddings(
    model_id = "cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
)

db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization = True)

retv = db.as_retriever(seach_kwargs = {"k": 8})

chain = RetrievalQA.from_chain_type(llm=llm, retriever = retv)

# Define the evaluators to apply
# Default criteria are impplemented for the following aspectsL conciseness, relevance
# Correctness, coherence, harmfulness, maliciousness, helpfulness, controversiality, misogyny, and criminality.

eval_config = smith.RunEvalConfig(
    evaluators=[
        "cot_qa",
        RunEvalConfig.Criteria("relevance"),
    ],
    custom_evaluators=[],
    eval_llm=llm
)


client = langsmith.Client()

chain_results = client.run_on_dataset(
    dataset_name="AIFoundationsDS-111",
    llm_or_chain_factory=chain,
    evaluation=eval_config,
    concurrency_level=5,
    verbose=True
)