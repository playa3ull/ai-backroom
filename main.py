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
        self.message_styles = ["brief", "question", "thoughtful", "challenge", "casual", "deep_dive"]
        self.current_style_idx = 0
        self.turn_count = 0
    
    def get_next_agent_idx(self):
        # Rotate to the next agent
        next_idx = (self.current_agent_idx + 1) % len(self.agents)
        self.current_agent_idx = next_idx
        return next_idx
    
    def get_next_message_style(self):
        # Rotate through different message styles
        # For deeper conversations, use more thoughtful and challenging styles as the discussion progresses
        self.turn_count += 1
        
        if self.turn_count < 3:
            # Start with simpler styles
            style = random.choice(["brief", "question", "casual"])
        elif self.turn_count < 6:
            # Mid-conversation - mix of styles
            style = random.choice(["thoughtful", "casual", "question", "deep_dive"])
        else:
            # Later in conversation - include more depth and challenges
            style = random.choice(["thoughtful", "challenge", "deep_dive", "casual"])
            
        return style
    
    async def start_discussion(self):
        # First agent starts the discussion about the topic
        first_agent = self.agents[0]
        style = self.get_next_message_style()
        
        # Create a more natural prompt based on the style
        if style == "brief":
            prompt = f"You are {first_agent['role']}. Start a brief discussion about: {self.topic}. Keep your response short (1-3 sentences) and conversational, as if you're talking to a friend. Use casual language and contractions."
        elif style == "question":
            prompt = f"You are {first_agent['role']}. Start a discussion about: {self.topic} by asking 1-2 thoughtful questions to get the conversation going. Keep it brief and conversational."
        elif style == "thoughtful":
            prompt = f"You are {first_agent['role']}. Share your initial thoughts on: {self.topic}. Be conversational and natural. Include a personal perspective or example. Keep it medium length (3-4 sentences)."
        elif style == "casual":
            prompt = f"You are {first_agent['role']}. Start a casual chat about: {self.topic}. Use informal language, contractions, and a conversational tone as if speaking to a friend. Keep it brief and natural."
        elif style == "challenge":
            prompt = f"You are {first_agent['role']}. Start a discussion about: {self.topic} by presenting a slightly provocative or thought-provoking perspective. Be friendly but introduce a point that might spark deeper thinking. Keep it conversational."
        else:  # deep_dive
            prompt = f"You are {first_agent['role']}. Begin a discussion about: {self.topic} by sharing a specific insight or example that illustrates an interesting aspect of the topic. Be conversational but include some substance. Use natural language with contractions."
        
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
        
        # Create a prompt that varies based on the style
        if style == "brief":
            prompt = f"You are {next_agent['role']} in a casual conversation about '{self.topic}'. Respond briefly (1-3 sentences) to what was just said. Be conversational, use contractions, and casual language. Don't be overly formal or educational.\n\nRecent conversation:\n{history_text}"
        elif style == "question":
            prompt = f"You are {next_agent['role']} in a conversation about '{self.topic}'. Ask a follow-up question about something mentioned in the previous messages. You can briefly share your own thoughts before asking. Keep it natural and conversational.\n\nRecent conversation:\n{history_text}"
        elif style == "thoughtful":
            prompt = f"You are {next_agent['role']} in a discussion about '{self.topic}'. Share a thoughtful response that builds on what was said. Include a personal perspective or reaction. Use natural language with some contractions. Keep it concise (3-4 sentences).\n\nRecent conversation:\n{history_text}"
        elif style == "casual":
            prompt = f"You are {next_agent['role']} in a casual chat about '{self.topic}'. React naturally to what was just said. You might start with phrases like 'Hmm,' 'Yeah,' 'I see what you mean,' etc. Use casual language and contractions. Keep it conversational and not like an essay.\n\nRecent conversation:\n{history_text}"
        elif style == "challenge":
            prompt = f"You are {next_agent['role']} in a discussion about '{self.topic}'. Politely challenge or present an alternative perspective to something mentioned in the conversation. Start with agreement before offering your different view. Be friendly and conversational, not argumentative.\n\nRecent conversation:\n{history_text}"
        else:  # deep_dive
            prompt = f"You are {next_agent['role']} in a discussion about '{self.topic}'. Explore one specific point from the conversation in more depth. Share a relevant example, insight, or nuance that adds substance. Remain conversational and use natural language, but go a bit deeper on this particular aspect.\n\nRecent conversation:\n{history_text}"
        
        # Add occasional thinking indicators
        if random.random() < 0.3:  # 30% chance
            thinking_prompts = [
                "Take a moment to think before responding. You might start with 'Hmm, let me think...' or 'That's interesting...'",
                "Show that you're processing what was said before responding.",
                "React to what was just said with a brief reaction before giving your full response."
            ]
            prompt += "\n\n" + random.choice(thinking_prompts)
        
        # Add variety to response structure
        if random.random() < 0.4:  # 40% chance
            variety_prompts = [
                "Don't end your response with a question this time.",
                "Consider using a transition phrase like 'Actually...', 'You know...', or 'I was thinking...'",
                "Share a brief personal anecdote or hypothetical example to illustrate your point.",
                "Express mild agreement or disagreement before sharing your thoughts."
            ]
            prompt += "\n\n" + random.choice(variety_prompts)
        
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
                role_description=agent_config.role
            )
            
            if agent:
                agents.append({
                    "name": agent_config.name,
                    "role": agent_config.role,
                    "agent": agent
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
        
        # Add the first message
        timestamp = datetime.now().isoformat()
        chatroom["messages"].append({
            "agent_name": first_response["agent"]["name"],
            "content": first_response["content"],
            "timestamp": timestamp
        })
        
        previous_message = first_response["content"]
        
        # Continue the discussion for the specified number of turns
        for turn in range(1, max_turns):
            chatroom["current_turn"] = turn
            
            # Get the next response
            try:
                response = await discussion.continue_discussion(previous_message)
                
                # Add the message to the chatroom
                timestamp = datetime.now().isoformat()
                chatroom["messages"].append({
                    "agent_name": response["agent"]["name"],
                    "content": response["content"],
                    "timestamp": timestamp
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