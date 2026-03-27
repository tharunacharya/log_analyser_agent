from autogen import AssistantAgent, UserProxyAgent

# Configure LLM to use Ollama (local LLM running in Codespace)
llm_config = {
    "config_list": [
        {
            "model": "phi3:mini",
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
        }
    ],
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
