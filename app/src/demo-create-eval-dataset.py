import os
from uuid import uuid4
from langsmith import Client

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<api-key>" # TODO: Input your api key

# create dataset for evaluation
dataset_inputs = [
    "Tell us about Oracle Cloud Infrastructure.",
    "Tell us about databases.",
    "Tell us about AI-optimized performance.",
    "Tell us about distributed cloud architecture."
    # ... add more as desired
]

# Output are provided to the evaluator, so it knows what to compare to
# Output are optional but recommended.

dataset_outputs = [
    {"must-mention": ["OCI", "performance"]},
    {"must-mention": ["data warehousing", "security"]},
    {"must-mention": ["GPU", "value"]},
    {"must-mention": ["multicloud", "flexibility"]},
]

client = Client()
dataset_name = "AIFoundationsDS-111"

# Storing inputs in a datset lets is
# run chains and LLMs over a shared set of examples

dataset = client.create_dataset(
    dataset_name = dataset_name,
    description= "AI(OCI) Foundations QA.",
)

client.create_examples(
    inputs = [{"question": q} for q in dataset_inputs],
    outputs=dataset_outputs,
    dataset_id = dataset.id,
)