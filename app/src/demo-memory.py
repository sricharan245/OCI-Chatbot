from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from langchain_community.llms import OCIGenAI

# In this demo we will explore using LangChain Memory to store chat history

# Step 1 - setup OCI Gen AI LLM
llm = OCIGenAI(
    model_id = "cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
    model_kwargs = {"max_tokens": 300}
)

# Step 2 - here we create a Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chaatboy who explain in steps."
        ),
        HumanMessagePromptTemplate.from_template(
            "{question}"
        )
    ]
)

# Step 3 - here we create a memory to remember our chat with the llm
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")


# Step 4 - Here we create a conversation chain using llm, prompt and memory

conversation = LLMChain(llm = llm, prompt = prompt, memory = memory)
# conversation = LLMChain(llm = llm, prompt = prompt, memory = summary_memory)

# Step 5 - here we invoke a chain. Notice thtat we just pass in the 'question' variables - 'chat_history' gets populated by memory
conversation.invoke({"question": "what is the capital of India"})

# Step 6 - here we print all the meassges in the memory

print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of the conversation is --> ", summary_memory.buffer)

#Stept 7 - here we ask another question

conversation.invoke({"question": "what is oci data science certification?"})

# Step 8 - here we print all the meassges in the memory again and see that our last question and response is printed

print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of the conversation is --> ", summary_memory.buffer)
