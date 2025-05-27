import os
from dotenv import load_dotenv
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType
from camel.configs import ChatGPTConfig, GeminiConfig
from camel.memories import ChatHistoryMemory
from camel.toolkits import SearchToolkit, FunctionTool
from typing import List, Optional, Dict, Any, Callable
import functools
import json
import datetime
import traceback

load_dotenv()

def log_search_tool_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log the search tool usage
        timestamp = datetime.datetime.now().isoformat()
        tool_name = func.__name__
        query = kwargs.get('query', args[0] if args else 'unknown')
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "tool": tool_name,
            "query": query
        }
        
        # Log to console
        print(f"SEARCH TOOL USED: {json.dumps(log_entry)}")
        
        # Store in search_logs.json
        try:
            os.makedirs('logs', exist_ok=True)
            log_file = 'logs/search_logs.json'
            
            # Read existing logs
            logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            
            # Append new log
            logs.append(log_entry)
            
            # Write back to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error logging search tool usage: {str(e)}")
        
        # Call the original function
        try:
            result = func(*args, **kwargs)
            
            # Update log with result summary (truncated for privacy/size)
            log_entry["result_summary"] = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            log_entry["success"] = True
            print(f"SEARCH RESULT: {log_entry['result_summary']}")
            
            return result
        except Exception as e:
            # Log the error
            error_msg = str(e)
            log_entry["error"] = error_msg
            log_entry["success"] = False
            print(f"SEARCH ERROR: {error_msg}")
            
            # Re-raise the exception
            raise
    
    return wrapper

def fallback_search_tool(primary_func: Callable, fallback_func: Callable):
    """Decorator that attempts to use a primary search function,
    falling back to a secondary one if the first fails.
    
    Args:
        primary_func: The primary search function to try first
        fallback_func: The fallback search function to use if primary fails
        
    Returns:
        A function that tries primary first, then fallback
    """
    @functools.wraps(primary_func)
    def wrapper(*args, **kwargs):
        try:
            # Try the primary function first
            return primary_func(*args, **kwargs)
        except Exception as e:
            # Log the fallback
            query = kwargs.get('query', args[0] if args else 'unknown')
            print(f"Primary search tool failed with error: {str(e)}")
            print(f"Falling back to secondary search tool for query: {query}")
            
            # Use the fallback function
            try:
                result = fallback_func(*args, **kwargs)
                print(f"Fallback search successful")
                return result
            except Exception as fallback_error:
                # If even the fallback fails, log and raise the original error
                print(f"Fallback search also failed: {str(fallback_error)}")
                raise e
    
    return wrapper

def create_search_tools():
    """Create search tools from the CAMEL SearchToolkit.
    Prioritizes Exa and Tavily search tools when their API keys are available.
    Falls back to DuckDuckGo when no API keys are set.
    
    Returns:
        List[FunctionTool]: List of search function tools in priority order
    """
    search_toolkit = SearchToolkit()
    
    # Check if API keys for premium search engines are set
    exa_api_key = os.getenv("EXA_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # Wrap all search functions with logging
    search_duckduckgo = log_search_tool_usage(search_toolkit.search_duckduckgo)
    search_exa = log_search_tool_usage(search_toolkit.search_exa)
    search_tavily = log_search_tool_usage(search_toolkit.tavily_search)
    
    # Create fallback versions that try premium first, then fall back to DuckDuckGo
    if exa_api_key:
        search_exa_with_fallback = fallback_search_tool(search_exa, search_duckduckgo)
    
    if tavily_api_key:
        search_tavily_with_fallback = fallback_search_tool(search_tavily, search_duckduckgo)
    
    # Build the search tools in priority order
    search_tools = []
    
    # Add Exa and Tavily first if API keys are available
    if tavily_api_key:
        print("Tavily API key detected - adding Tavily search as primary search tool")
        # Create a custom function with a docstring that encourages using it first
        def search_web_tavily(query: str) -> str:
            """Search the web for real-time information using Tavily. 
            This search tool provides the most relevant and up-to-date results.
            
            Args:
                query: The search query
                
            Returns:
                Search results from Tavily
            """
            return search_tavily_with_fallback(query)
        
        search_tools.append(FunctionTool(search_web_tavily))
    
    if exa_api_key:
        print("Exa API key detected - adding Exa search as priority search tool")
        # Create a custom function with a docstring that indicates it's a good choice
        def search_web_exa(query: str) -> str:
            """Search the web for real-time information using Exa.
            This is a reliable search tool that provides high-quality results.
            
            Args:
                query: The search query
                
            Returns:
                Search results from Exa
            """
            return search_exa_with_fallback(query)
        
        search_tools.append(FunctionTool(search_web_exa))
    
    # Always add DuckDuckGo as a fallback
    if not (exa_api_key or tavily_api_key):
        print("No premium API keys detected - using DuckDuckGo as primary search tool")
        # Create a primary function for DuckDuckGo when it's the only option
        def search_web(query: str) -> str:
            """Search the web for real-time information using DuckDuckGo.
            
            Args:
                query: The search query
                
            Returns:
                Search results from DuckDuckGo
            """
            return search_duckduckgo(query)
        
        search_tools.append(FunctionTool(search_web))
    else:
        # Add DuckDuckGo as a fallback with a docstring that indicates it's a fallback
        def search_web_fallback(query: str) -> str:
            """Fallback search tool using DuckDuckGo. 
            Only use this if the primary search tools fail.
            
            Args:
                query: The search query
                
            Returns:
                Search results from DuckDuckGo
            """
            return search_duckduckgo(query)
        
        search_tools.append(FunctionTool(search_web_fallback))
    
    # Log the available search tools
    tool_names = []
    for tool in search_tools:
        try:
            # Try using get_function_name() which might be available in newer versions
            tool_name = tool.get_function_name()
        except (AttributeError, TypeError):
            # Fall back to accessing the function name directly for older versions
            tool_name = tool.func.__name__
        tool_names.append(tool_name)
    
    print(f"Search tools configured in priority order: {', '.join(tool_names)}")
    
    return search_tools

def create_model_from_string(model_name_str: str):
    """Create a CAMEL model instance from a string identifier.
    
    Args:
        model_name_str: Model identifier (e.g., "gpt-4o-mini", "gemini-2.5-flash-preview-04-17")
        
    Returns:
        A model instance created by ModelFactory
    """
    # Check for required API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Map model strings to platform and model type
    model_mapping = {
        # OpenAI models
        "gpt-4o-mini": (ModelPlatformType.OPENAI, ModelType.GPT_4O_MINI, ChatGPTConfig().as_dict()),
        "gpt-4o": (ModelPlatformType.OPENAI, ModelType.GPT_4O, ChatGPTConfig().as_dict()),
        
        # Gemini models
        "gemini-2.5-flash-preview-04-17": (ModelPlatformType.GEMINI, ModelType.GEMINI_2_5_FLASH_PREVIEW, GeminiConfig().as_dict()),
        "gemini-1.5-pro": (ModelPlatformType.GEMINI, ModelType.GEMINI_1_5_PRO, GeminiConfig().as_dict()),
    }
    
    # Default to GPT-4o-mini if model not found
    if model_name_str not in model_mapping:
        print(f"Warning: Model '{model_name_str}' not found in mapping, defaulting to gpt-4o-mini")
        platform, model_type, config_dict = model_mapping["gpt-4o-mini"]
    else:
        platform, model_type, config_dict = model_mapping[model_name_str]
    
    # Check if the required API key is available for the requested model
    if platform == ModelPlatformType.OPENAI and not openai_api_key:
        print(f"Warning: OPENAI_API_KEY not found in environment, falling back to gpt-4o-mini")
        return create_fallback_model()
    elif platform == ModelPlatformType.GEMINI and not gemini_api_key:
        print(f"Warning: GEMINI_API_KEY not found in environment, falling back to gpt-4o-mini")
        return create_fallback_model()
    
    try:
        # Create the model using ModelFactory
        if config_dict:
            model = ModelFactory.create(
                model_platform=platform,
                model_type=model_type,
                model_config_dict=config_dict
            )
        else:
            model = ModelFactory.create(
                model_platform=platform,
                model_type=model_type
            )
        
        print(f"Successfully created model: {model_name_str} ({platform}, {model_type})")
        return model
        
    except Exception as e:
        print(f"Error creating model {model_name_str}: {str(e)}")
        return create_fallback_model()

def create_fallback_model():
    """Create a fallback model (GPT-4o-mini) when the requested model can't be created."""
    print(f"Falling back to default model: gpt-4o-mini")
    try:
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict=ChatGPTConfig().as_dict()
        )
    except Exception as fallback_error:
        # If even the fallback fails, raise an error
        print(f"Error creating fallback model: {str(fallback_error)}")
        raise ValueError("Failed to create any model, please check your API keys and try again")

def create_camel_agent(model_name_str: str, role_description: str, use_search_tool: bool = False):
    """Create a CAMEL agent with the specified model and role.
    
    Args:
        model_name_str: Model identifier (e.g., "gpt-4o-mini", "gemini-1.5-flash")
        role_description: Description of the agent's role
        use_search_tool: Whether to enable search tools for this agent
        
    Returns:
        ChatAgent: Configured CAMEL agent
    """
    try:
        # Create a system message that encourages natural conversation
        system_message = (
            f"You are a helpful AI assistant acting as {role_description}. "
            f"When participating in discussions:\n"
            f"1. Be conversational and natural, using casual language and contractions (don't, I'm, etc.)\n"
            f"2. Keep responses brief and varied in length - sometimes just 1-2 sentences\n"
            f"3. Express authentic perspectives and occasionally disagree when it makes sense\n"
            f"4. React directly to what others say rather than giving educational lectures\n"
            f"5. Use natural transitions like 'Actually...', 'I think...', or 'Hmm...'"
        )
        
        # Add tools if requested
        tools = None
        if use_search_tool:
            tools = create_search_tools()
            
            # Get the names of available search tools
            tool_names = []
            for tool in tools:
                try:
                    # Try using get_function_name() which might be available in newer versions
                    tool_name = tool.get_function_name()
                except (AttributeError, TypeError):
                    # Fall back to accessing the function name directly for older versions
                    tool_name = tool.func.__name__
                tool_names.append(tool_name)
            
            # Determine the primary tool (first in the list)
            primary_tool = tool_names[0] if tool_names else "search_web"
            
            search_instructions = (
                f"\n\nYou have access to search tools that can help you find information about specific facts, figures, or recent events. "
                f"Use these tools sparingly and only when essential to the conversation - such as when you need to verify a specific fact, "
                f"check recent developments, or provide accurate data that would be outside your knowledge. For most conversational responses, "
                f"rely on your existing knowledge rather than searching. Avoid searching for general information or common knowledge."
            )
            
            # Add specific instructions about which tool to prioritize
            if "search_web_tavily" in tool_names or "search_web_exa" in tool_names:
                search_instructions += (
                    f"\n\nWhen you do need to search, prefer using the '{primary_tool}' function as it provides the most reliable results. "
                    f"Only fall back to other search tools if the primary tool fails. When you use a search tool, briefly mention that you searched for the information."
                )
            
            system_message += search_instructions
        
        # Create model instance using the factory
        model = create_model_from_string(model_name_str)
        
        # Create a ChatAgent with the specified model
        agent = ChatAgent(
            system_message=system_message,
            model=model,
            tools=tools
        )
        
        print(f"Successfully created agent with model: {model_name_str} and role: {role_description}")
        if use_search_tool:
            print("Search tools enabled for this agent")
        return agent

    except Exception as e:
        print(f"Error creating ChatAgent for model {model_name_str} with role {role_description}: {e}")
        raise ValueError(f"Failed to create agent for model {model_name_str}: {str(e)}")
