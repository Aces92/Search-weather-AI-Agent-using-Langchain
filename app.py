import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from main import create_research_agent

# ======================
# Load Agent
# ======================
@st.cache_resource
def load_agent():
    return create_research_agent()

agent = load_agent()

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
        }
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.3rem;
        }
        .subtitle {
            color: #a0aec0;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ======================
# Sidebar
# ======================
with st.sidebar:
    st.title("⚙️ Settings")
    
    show_reasoning = st.toggle("Show Agent Reasoning", value=False)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Available Tools")
    st.markdown("- 🔍 **Tavily Search** (Web search)")
    st.markdown("- 🌤️ **Weather Tool** (Current weather)")
    
    st.markdown("---")
    st.caption("Built with LangChain + Groq + Streamlit")

# ======================
# Main Chat Interface
# ======================
st.markdown('<p class="main-title">🤖 AI Research Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask me anything — I can search the web and check the weather!</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "reasoning" in message and show_reasoning:
            with st.expander("🧠 View Agent Reasoning"):
                st.markdown(message["reasoning"])

# Chat input
if prompt := st.chat_input("Ask me something..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.invoke({
                    "messages": [{"role": "user", "content": prompt}]
                })

                final_answer = response["messages"][-1].content
                st.markdown(final_answer)

                # Build reasoning steps
                reasoning_steps = []
                for msg in response["messages"]:
                    if isinstance(msg, HumanMessage):
                        reasoning_steps.append(f"**🧑 Human:** {msg.content}")
                    elif isinstance(msg, AIMessage):
                        if msg.content:
                            reasoning_steps.append(f"**🤖 AI:** {msg.content}")
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                reasoning_steps.append(
                                    f"**➜ Tool Call:** `{tool_call['name']}` with `{tool_call['args']}`"
                                )
                    elif isinstance(msg, ToolMessage):
                        reasoning_steps.append(f"**🛠️ Tool Result ({msg.name}):**\n{msg.content}")

                reasoning_text = "\n\n".join(reasoning_steps)

                if show_reasoning:
                    with st.expander("🧠 View Agent Reasoning", expanded=True):
                        st.markdown(reasoning_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "reasoning": reasoning_text
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")