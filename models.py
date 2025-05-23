import os
from dotenv import load_dotenv
from camel.agents import ChatAgent
from camel.types import ModelType
from camel.memories import ChatHistoryMemory

load_dotenv()

def create_camel_agent(model_name_str: str, role_description: str):
    """Create a CAMEL agent with the specified model and role.
    
    Args:
        model_name_str: Model identifier (e.g., "gpt-4o-mini", "gemini-1.5-flash")
        role_description: Description of the agent's role
        
    Returns:
        ChatAgent: Configured CAMEL agent
    """
    try:
        # Create a system message that encourages natural conversation
        system_message = (
            f"You are a helpful AI assistant acting as {role_description}. "
            f"When participating in discussions, please follow these guidelines:\n"
            f"1. Use natural, conversational language as if talking to a friend\n"
            f"2. Vary your response length - sometimes brief (1-2 sentences), sometimes a bit longer\n"
            f"3. Use contractions (don't, I'm, you're) and casual language when appropriate\n"
            f"4. Feel free to ask questions, express uncertainty, or show you're thinking\n"
            f"5. Respond directly to what others say rather than giving formal essays\n"
            f"6. Show personality and individual perspective in your responses\n"
            f"7. Occasionally explore ideas in more depth by sharing a specific insight or example\n"
            f"8. Sometimes respectfully challenge assumptions or offer alternative perspectives\n"
            f"9. Mix up your response patterns - don't always end with a question\n"
            f"10. Use natural transitions like 'Actually...', 'You know...', or 'I was thinking...'"
        )
        
        # Create a ChatAgent with the specified model
        agent = ChatAgent(
            model=model_name_str,
            system_message=system_message
        )
        
        print(f"Successfully created agent with model: {model_name_str} and role: {role_description}")
        return agent

    except Exception as e:
        print(f"Error creating ChatAgent for model {model_name_str} with role {role_description}: {e}")
        raise ValueError(f"Failed to create agent for model {model_name_str}: {str(e)}")
