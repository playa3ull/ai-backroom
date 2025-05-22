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
from camel.societies import RolePlaying
from camel.types import TaskType
import asyncio
import json
from dotenv import load_dotenv
from models import create_camel_agent

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

# Model type mapping is now imported from models.py

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chatroom/create")
async def create_chatroom(config: ChatroomConfig):
    # Generate a unique ID for the chatroom
    import uuid
    chatroom_id = str(uuid.uuid4())
    
        # Initialize RolePlaying society for each pair of agents
    societies = []
    agents = []
    
    # We need at least 2 agents for RolePlaying
    if len(config.agents) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 agents for RolePlaying discussion")
    
    # Create pairs of agents for RolePlaying societies
    for i in range(0, len(config.agents), 2):
        if i + 1 >= len(config.agents):
            break
            
        user_config = config.agents[i]
        assistant_config = config.agents[i + 1]
        
        try:
            # Create model instances for both agents
            user_model = create_camel_agent(model_name_str=user_config.model_type, role_description=user_config.role)
            assistant_model = create_camel_agent(model_name_str=assistant_config.model_type, role_description=assistant_config.role)
            
            if user_model and assistant_model:
                # Create RolePlaying society
                society = RolePlaying(
                    task_prompt=config.topic.title,
                    user_role_name=user_config.role,
                    assistant_role_name=assistant_config.role,
                    user_agent_kwargs={"model": user_model},
                    assistant_agent_kwargs={"model": assistant_model}
                )
                
                societies.append(society)
                agents.extend([{
                    "name": user_config.name,
                    "role": user_config.role,
                    "agent": user_model
                }, {
                    "name": assistant_config.name,
                    "role": assistant_config.role,
                    "agent": assistant_model
                }])
            else:
                print(f"Warning: Could not create model instances for the agent pair.")
        except Exception as e:
            return {"error": f"Failed to initialize RolePlaying society: {str(e)}"}
    
    # Store chatroom configuration with societies
    active_chatrooms[chatroom_id] = {
        "config": config.dict(),
        "agents": agents,
        "societies": societies,
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
    agents = chatroom["agents"]
    
    if len(agents) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 agents for a discussion")
    
    # Initialize the discussion with the topic
    topic = config["topic"]
    initial_message = f"Let's discuss the topic: {topic['title']}"
    
    # Add the initial message
    from datetime import datetime
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
    from datetime import datetime
    
    chatroom = active_chatrooms[chatroom_id]
    config = chatroom["config"]
    societies = chatroom["societies"]
    max_turns = config["max_turns"]
    
    input_msg = None  # Initialize input_msg before the loop
    # Run the discussion for the specified number of turns
    for turn in range(max_turns):
        chatroom["current_turn"] = turn + 1
        
        for society in societies:
            try:
                # Initialize chat for this turn if it's the first turn
                if turn == 0:
                    input_msg = society.init_chat()
                    if input_msg:
                        # Add the initial message from assistant
                        timestamp = datetime.now().isoformat()
                        chatroom["messages"].append({
                            "agent_name": society.assistant_role_name,
                            "content": input_msg.content,
                            "timestamp": timestamp
                        })
                # Get responses from both agents in the society
                assistant_response, user_response = await society.astep(input_msg)
                # After the first turn, update input_msg for the next turn
                input_msg = user_response if user_response else assistant_response
                
                # Add messages to the chatroom
                timestamp = datetime.now().isoformat()
                if assistant_response and assistant_response.msgs:
                    chatroom["messages"].append({
                        "agent_name": society.assistant_role_name,
                        "content": assistant_response.msgs[0].content,
                        "timestamp": timestamp
                    })
                
                if user_response and user_response.msgs:
                    chatroom["messages"].append({
                        "agent_name": society.user_role_name,
                        "content": user_response.msgs[0].content,
                        "timestamp": timestamp
                    })
                
                # Update input message for next turn
                if assistant_response and assistant_response.msgs:
                    input_msg = assistant_response.msgs[0]
                
                # Check for task completion
                if user_response and "CAMEL_TASK_DONE" in user_response.msgs[0].content:
                    break
                
                # Small delay between turns
                await asyncio.sleep(1)
                
            except Exception as e:
                error_msg = f"Error in society discussion: {str(e)}"
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