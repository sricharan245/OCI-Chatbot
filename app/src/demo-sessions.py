from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OCIGenAI


# In this demo we will explore using streamlit to store chat messages

# Step 1 - setup OCI Gen AI LLM
llm = OCIGenAI(
    model_id = "cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaazjn75n5agh4vmgtht3jyabvyxnfdlylgrbspbh2dmrrulkjlzpsa",
    model_kwargs = {"max_tokens": 300}
)

# Step 2 - here we create a history with a key "Chat Messages"

# StreamlitChatMessageHistory will store messages in streamlit session state at the specified key=
# A given StreamlitChatMessageHistory will  not be presisted or shared across the users

history = StreamlitChatMessageHistory(key="chat_messages")

# Step 3 - here we create  memory object

memory = ConversationBufferMemory(chat_memory=history)

# Step 4 - here we create a template and prompt to accept a question

template = """You are an AI chatbot having a conversation witha  human.
Human: {human_input}
AI: """

prompt = PromptTemplate(input_variables=['human_input'], template=template)

#Step 5 - here we create a chain object

llm_chain = LLMChain(llm=llm, prompt = prompt, memory = memory)

#Step 6 = here we ise Streamlit to print all messages in the memory, ctreat text input, run chain and 
# the question and response is automatically put in the StreamlitChatMessageHistory

import streamlit as st

st.title("Welcome to the ChatBot")
for msg in history.messages:
    print(msg.type)
    st.chat_message(msg.type).write(msg.content)

if x := st.chat_input():
    st.chat_message('human').write(x)

    # As usual, new messages are added to StreamlitChatMessageHistory when the chain is called
    response = llm_chain.run(x)
    st.chat_message("ai").write(response)