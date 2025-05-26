import os
from dotenv import load_dotenv
from camel.agents import ChatAgent
from camel.types import ModelType
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
                f"\n\nYou have access to search tools that can help you find information. "
                f"When you need to look up information that might be outside your knowledge, "
                f"use these tools. Don't make up information - if you're unsure, "
                f"use the search tools to find accurate information."
            )
            
            # Add specific instructions about which tool to prioritize
            if "search_web_tavily" in tool_names or "search_web_exa" in tool_names:
                search_instructions += (
                    f"\n\nIMPORTANT: You have multiple search tools available. "
                    f"Always prefer using the '{primary_tool}' function first as it provides the most reliable results. "
                    f"Only fall back to other search tools if the primary tool fails or doesn't return useful results. "
                    f"When you use a search tool, explicitly mention that you searched for information."
                )
            
            system_message += search_instructions
        
        # Create a ChatAgent with the specified model
        agent = ChatAgent(
            model=model_name_str,
            system_message=system_message,
            tools=tools
        )
        
        print(f"Successfully created agent with model: {model_name_str} and role: {role_description}")
        if use_search_tool:
            print("Search tools enabled for this agent")
        return agent

    except Exception as e:
        print(f"Error creating ChatAgent for model {model_name_str} with role {role_description}: {e}")
        raise ValueError(f"Failed to create agent for model {model_name_str}: {str(e)}")
