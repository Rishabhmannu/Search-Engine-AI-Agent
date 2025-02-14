import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# 1. Create Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(
    name="Search",
    region="us-en",
    safesearch="Off",
    time="y",
    max_results=3
)

# 2. Streamlit UI
st.title("🔎 LangChain - Chat with search (Improved)")

"""
In this example, we instruct the agent to search only until it has 
enough information, then provide a concise final answer.
"""

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "Hi, I'm a chatbot with access to search tools. Ask me anything!"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input(placeholder="Ask me something, e.g. 'What is machine learning?'")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 3. Supply System-Like Instructions via the Agent Prefix
    system_instructions = (
        "You are a concise, knowledgeable assistant. "
        "Use the provided tools ONLY IF absolutely necessary to answer the user's query. "
        "Once you have enough information, STOP searching and compile a final, concise answer. "
        "Do not keep repeating the same queries."
    )

    # 4. Create ChatGroq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True
        # No `system_message` here; ChatGroq doesn't support that param
    )

    # 5. Initialize Agent with a prompt prefix
    #    max_iterations=3 ensures the agent doesn’t get stuck searching forever
    agent = initialize_agent(
        tools=[search, arxiv, wiki],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="force",
        verbose=False,  # set True if you want more logs in your terminal
        agent_kwargs={
            "prefix": system_instructions,  
        }
    )

    # 6. Run the Agent
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.invoke(
            {"input": user_input},
            callbacks=[st_cb]
        )["output"]

        # Save to session and display
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
