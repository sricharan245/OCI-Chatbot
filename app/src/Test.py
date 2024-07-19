from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


from langchain_community.llms import OCIGenAI

from oci.config import from_file

# Step 1 - authenticate using "DEFAULT" profile
config = from_file(file_location="~/.oci/config")

# Step 2 - Setup OCI Generative AI llm
llm = OCIGenAI(
    # model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyawk6mgunzodenakhkuwxanvt6wo3jcpf72ln52dymk4wq",
    model_id = "cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
    model_kwargs = {"max_tokens": 200}
)

response = llm.invoke("Tell me one fact about space", temperature=0.7)
print("Case 1 Response:", response)

#############
# Step 3 - use string prompt to accept text input. Here we create a template and declare a input variable {human_input}

# String prompt

template = """You are a chatbot having a conversation with a human.
Human: {human_input} + {city}
"""

# step 4 - Here we create a Prompt using the template

prompt2 = PromptTemplate(input_variables=["human_input", "city"], template=template)

prompt_val = prompt2.invoke({"human_input":"Tell us in a exciting tone about", "city": "Las Vegas"})
print ("Prompt String is -> ")
print (prompt_val.to_string())

#Step 5 - here we declare a chain that beigns with a prompt, next llm and finally output parser

chain = prompt2 | llm

# Step 6 - Next we invoke a chain and provide input question

response = chain.invoke({"human_input":"Tell us in a exciting tone about", "city": "Las Vegas"})

#Step 7 - print the prompt and response from the llm

print("Case2 Response - > ", response)
