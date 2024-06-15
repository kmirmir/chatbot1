import dotenv
dotenv.load_dotenv()

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

import streamlit as st
import time

st.title("AI Chatbot powered by OpenAI")
st.caption("언어모델 : gpt-4o")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # LLM
        llm = ChatOpenAI()

        # Prompt
        llm_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
        # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation = LLMChain(llm=llm, prompt=llm_prompt, verbose=True, memory=memory)

        result = conversation({"question": prompt})
        for chunk in result['text'].split():
            full_response += chunk + " "
            time.sleep(0.05)

            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})