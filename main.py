import os
import uvicorn
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import camel
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import TaskType
import asyncio
import json
from dotenv import load_dotenv
from models import create_camel_agent
from datetime import datetime
import random

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Chatroom")

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Define models
class Topic(BaseModel):
    title: str

class AgentConfig(BaseModel):
    name: str
    model_type: str
    role: str
    use_search_tool: bool = False
    
class ChatroomConfig(BaseModel):
    topic: Topic
    agents: List[AgentConfig]
    max_turns: int = 10

class ChatMessage(BaseModel):
    agent_name: str
    content: str
    timestamp: Optional[str] = None

# Store for active chatrooms
active_chatrooms: Dict[str, Dict] = {}

# Simple Discussion class to replace RolePlaying
class SimpleDiscussion:
    def __init__(self, topic, agents):
        self.topic = topic
        self.agents = agents
        self.current_agent_idx = 0
        self.conversation_history = []
        self.message_styles = ["standard", "question", "challenge", "deep_dive"]
        self.current_style_idx = 0
        self.turn_count = 0
        self.last_disagreement = -3  # Track when the last disagreement happened
    
    def get_next_agent_idx(self):
        # Rotate to the next agent
        next_idx = (self.current_agent_idx + 1) % len(self.agents)
        self.current_agent_idx = next_idx
        return next_idx
    
    def get_next_message_style(self):
        # Simplified style selection
        self.turn_count += 1
        
        # Occasionally choose challenge style to ensure healthy disagreement
        if random.random() < 0.25 and (self.turn_count - self.last_disagreement) >= 3:
            self.last_disagreement = self.turn_count
            return "challenge"
        
        # Choose random style based on conversation progress
        if self.turn_count < 3:
            return random.choice(["standard", "question"])
        else:
            return random.choice(self.message_styles)
    
    async def start_discussion(self):
        # First agent starts the discussion about the topic
        first_agent = self.agents[0]
        style = self.get_next_message_style()
        
        # Create a simplified prompt based on the style
        if style == "question":
            prompt = f"You are {first_agent['role']}. Start a discussion about: {self.topic} by asking 1-2 thoughtful but casual questions."
        elif style == "challenge":
            prompt = f"You are {first_agent['role']}. Start a discussion about: {self.topic} by presenting a thought-provoking perspective. Be conversational, not academic."
        elif style == "deep_dive":
            prompt = f"You are {first_agent['role']}. Begin a discussion about: {self.topic} by sharing a specific insight or example, but keep it casual and brief."
        else:  # standard
            prompt = f"You are {first_agent['role']}. Start a brief, casual chat about: {self.topic}. Speak naturally as if talking to a friend."
        
        # Add instruction to be authentic
        prompt += "\n\nBe authentic and conversational - avoid sounding like you're giving a lecture."
        
        response = await first_agent["agent"].astep(prompt)
        content = response.msgs[0].content if response.msgs else "I'd like to discuss this topic."
        
        # Store in conversation history
        self.conversation_history.append({
            "agent": first_agent["name"],
            "content": content
        })
        
        return {
            "agent": first_agent,
            "content": content
        }
    
    async def continue_discussion(self, previous_message):
        # Get the next agent to respond
        next_idx = self.get_next_agent_idx()
        next_agent = self.agents[next_idx]
        style = self.get_next_message_style()
        
        # Get the last few messages for context (up to 4)
        recent_history = self.conversation_history[-4:] if len(self.conversation_history) > 4 else self.conversation_history
        history_text = "\n".join([f"{msg['agent']}: {msg['content']}" for msg in recent_history])
        
        # Create a streamlined prompt based on the style
        base_prompt = f"You are {next_agent['role']} in a discussion about '{self.topic}'."
        
        if style == "question":
            prompt = f"{base_prompt} Ask a follow-up question about something mentioned in the previous messages. Keep it casual and brief."
        elif style == "challenge":
            prompt = f"{base_prompt} Present an alternative perspective to something mentioned in the conversation. Start with a phrase like 'I'm not sure about that...' or 'I see it differently...'."
        elif style == "deep_dive":
            prompt = f"{base_prompt} Explore one specific point from the conversation in more depth, but keep it conversational, not educational."
        else:  # standard
            prompt = f"{base_prompt} Respond briefly and naturally to what was just said, as if chatting with a friend."
        
        prompt += f"\n\nRecent conversation:\n{history_text}"
        
        # Add search tool instructions if available
        if next_agent.get('use_search_tool', False):
            # Randomly determine if we should strongly encourage search in this turn
            should_encourage_search = random.random() < 0.4  # 40% chance to strongly encourage search
            
            if should_encourage_search:
                # Only encourage search for specific fact-checking scenarios
                prompt += f"\n\nIf you need to verify a specific fact or figure about '{self.topic}' in this response, you may use your search tool. Only search if truly necessary for accuracy."
            else:
                # Default instruction discourages unnecessary searching
                prompt += f"\n\nYou have access to search tools, but only use them when absolutely necessary to verify specific facts. For most responses, rely on your existing knowledge."
        
        # Add authenticity reminder
        prompt += "\n\nBe authentic in your response - it's fine to have a different perspective or opinion."
        
        # Get response from the agent
        response = await next_agent["agent"].astep(prompt)
        content = response.msgs[0].content if response.msgs else "I'd like to add to this discussion."
        
        # Store in conversation history
        self.conversation_history.append({
            "agent": next_agent["name"],
            "content": content
        })
        
        return {
            "agent": next_agent,
            "content": content
        }

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chatroom/create")
async def create_chatroom(config: ChatroomConfig):
    # Generate a unique ID for the chatroom
    import uuid
    chatroom_id = str(uuid.uuid4())
    
    # Initialize agents
    agents = []
    
    # We need at least 2 agents for a discussion
    if len(config.agents) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 agents for a discussion")
    
    # Create agents
    for agent_config in config.agents:
        try:
            # Create agent instance
            agent = create_camel_agent(
                model_name_str=agent_config.model_type, 
                role_description=agent_config.role,
                use_search_tool=agent_config.use_search_tool
            )
            
            if agent:
                agents.append({
                    "name": agent_config.name,
                    "role": agent_config.role,
                    "model_type": agent_config.model_type,
                    "agent": agent,
                    "use_search_tool": agent_config.use_search_tool
                })
            else:
                print(f"Warning: Could not create agent instance for {agent_config.name}.")
        except Exception as e:
            return {"error": f"Failed to initialize agent {agent_config.name}: {str(e)}"}
    
    # Create a simple discussion
    discussion = SimpleDiscussion(
        topic=config.topic.title,
        agents=agents
    )
    
    # Store chatroom configuration
    active_chatrooms[chatroom_id] = {
        "config": config.dict(),
        "agents": agents,
        "discussion": discussion,
        "messages": [],
        "current_turn": 0
    }
    
    # Initialize the discussion with the topic
    topic = config.topic
    initial_message = f"Let's discuss the topic: {topic.title}"
    
    # Add the initial message
    timestamp = datetime.now().isoformat()
    active_chatrooms[chatroom_id]["messages"].append({
        "agent_name": "System",
        "content": initial_message,
        "timestamp": timestamp
    })
    
    # Start the discussion asynchronously
    asyncio.create_task(run_discussion(chatroom_id))
    
    return {"chatroom_id": chatroom_id}

@app.get("/api/chatroom/{chatroom_id}")
async def get_chatroom(chatroom_id: str):
    if chatroom_id not in active_chatrooms:
        raise HTTPException(status_code=404, detail="Chatroom not found")
    
    chatroom = active_chatrooms[chatroom_id]
    return {
        "config": chatroom["config"],
        "messages": chatroom["messages"],
        "current_turn": chatroom["current_turn"]
    }

@app.post("/api/chatroom/{chatroom_id}/start")
async def start_discussion(chatroom_id: str):
    if chatroom_id not in active_chatrooms:
        raise HTTPException(status_code=404, detail="Chatroom not found")
    
    chatroom = active_chatrooms[chatroom_id]
    config = chatroom["config"]
    
    # Check if discussion is already started
    if len(chatroom["messages"]) > 0:
        return {"status": "Discussion already started"}
    
    # Initialize the discussion with the topic
    topic = config["topic"]
    initial_message = f"Let's discuss the topic: {topic['title']}"
    
    # Add the initial message
    timestamp = datetime.now().isoformat()
    chatroom["messages"].append({
        "agent_name": "System",
        "content": initial_message,
        "timestamp": timestamp
    })
    
    # Start the discussion asynchronously
    asyncio.create_task(run_discussion(chatroom_id))
    
    return {"status": "Discussion started"}

async def run_discussion(chatroom_id: str):
    chatroom = active_chatrooms[chatroom_id]
    config = chatroom["config"]
    discussion = chatroom["discussion"]
    max_turns = config["max_turns"]
    
    try:
        # Start the discussion
        first_response = await discussion.start_discussion()
        
        # Check if search was used in the response
        search_used = False
        if hasattr(first_response["agent"]["agent"], "info") and first_response["agent"]["agent"].info:
            if "tool_calls" in first_response["agent"]["agent"].info and first_response["agent"]["agent"].info["tool_calls"]:
                for tool_call in first_response["agent"]["agent"].info["tool_calls"]:
                    if "search" in tool_call.name.lower():
                        search_used = True
                        break
        
        # Add the first message
        timestamp = datetime.now().isoformat()
        chatroom["messages"].append({
            "agent_name": first_response["agent"]["name"],
            "model_type": first_response["agent"]["model_type"],
            "content": first_response["content"],
            "timestamp": timestamp,
            "search_used": search_used
        })
        
        previous_message = first_response["content"]
        
        # Continue the discussion for the specified number of turns
        for turn in range(1, max_turns):
            chatroom["current_turn"] = turn
            
            # Get the next response
            try:
                response = await discussion.continue_discussion(previous_message)
                
                # Check if search was used in the response
                search_used = False
                if hasattr(response["agent"]["agent"], "info") and response["agent"]["agent"].info:
                    if "tool_calls" in response["agent"]["agent"].info and response["agent"]["agent"].info["tool_calls"]:
                        for tool_call in response["agent"]["agent"].info["tool_calls"]:
                            if "search" in tool_call.name.lower():
                                search_used = True
                                break
                
                # Also check content for search indicators
                content_indicates_search = (
                    "I searched for" in response["content"] or
                    "According to my search" in response["content"] or
                    "Based on my search" in response["content"] or
                    "search results show" in response["content"]
                )
                
                search_used = search_used or content_indicates_search
                
                # Add the message to the chatroom
                timestamp = datetime.now().isoformat()
                chatroom["messages"].append({
                    "agent_name": response["agent"]["name"],
                    "model_type": response["agent"]["model_type"],
                    "content": response["content"],
                    "timestamp": timestamp,
                    "search_used": search_used
                })
                
                # Update previous message for next turn
                previous_message = response["content"]
                
                # Small delay between turns
                await asyncio.sleep(1)
                
            except Exception as e:
                error_msg = f"Error in discussion: {str(e)}"
                chatroom["messages"].append({
                    "agent_name": "System",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
    
    except Exception as e:
        error_msg = f"Error starting discussion: {str(e)}"
        chatroom["messages"].append({
            "agent_name": "System",
            "content": error_msg,
            "timestamp": datetime.now().isoformat()
        })
    
    # Add a conclusion message
    chatroom["messages"].append({
        "agent_name": "System",
        "content": "The discussion has concluded. Thank you for participating!",
        "timestamp": datetime.now().isoformat()
    })

@app.get("/chatroom/{chatroom_id}", response_class=HTMLResponse)
async def view_chatroom(request: Request, chatroom_id: str):
    if chatroom_id not in active_chatrooms:
        raise HTTPException(status_code=404, detail="Chatroom not found")
    
    return templates.TemplateResponse("chatroom.html", {
        "request": request,
        "chatroom_id": chatroom_id
    })

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)