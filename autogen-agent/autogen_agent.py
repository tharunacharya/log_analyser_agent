import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent

# Load environment variables from parent .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configure LLM to use Groq (OpenAI-compatible API)
llm_config = {
    "model": "llama-3.3-70b-versatile",
    "api_key": os.getenv("GROQ_API_KEY"),
    "base_url": "https://api.groq.com/openai/v1",
}

# Task 1: Simple two-agent system
# Assistant agent responds to questions using Groq LLM
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant. Answer questions clearly and concisely.",
)

# User proxy agent sends a question (human_input_mode=NEVER for automated run)
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False,
)

# Start the conversation - user agent asks, assistant responds
if __name__ == "__main__":
    print("\nStarting AutoGen Agent Conversation...\n")
    user_proxy.initiate_chat(
        assistant,
        message="What are the most common causes of database connection timeouts in production environments?",
    )
