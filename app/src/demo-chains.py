from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a very knowledgeable scientist who provides accurate and eloquent answers to scientific questions."),
        ("human", "{question}")
    ]
)

#Step 3 - Using legacy LLMChain object and passing llm object, prompt template and output parser

chain  = LLMChain(llm = llm, prompt = prompt, output_parser = StrOutputParser())
response = chain.invoke({"question":"What are basic elements of a matter?"})
print("Response from legacy chain")
print(response)

# #########
print("------")
#Step 4 - Using LangChain Expression Language (LCEL) LLMChain object and 
# creating runnable chain which contain llm object, prompt template and output parser
runnable  = prompt | llm | StrOutputParser()

response = runnable.invoke({"question":"What are basic elements of a matter?"})
print("Response from LCEL chain")
print(response)
