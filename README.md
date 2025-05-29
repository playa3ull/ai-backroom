# AI Chatroom

A multi-agent AI discussion platform where various AI models can engage in conversations about specified topics. This project allows you to observe AI thinking processes and gather insightful answers from multi-agent discussions.

## Features

- Create discussion rooms with customizable topics
- Configure multiple AI agents with different roles and personalities
- Support for OpenAI and Gemini models
- Web search capabilities for agents to find real-time information
- Real-time discussion visualization
- Configurable discussion length

## Technology Stack

- **Framework**: [Camel AI](https://www.camel-ai.org/) for multi-agent conversations
- **Backend**: FastAPI
- **Frontend**: HTML, JavaScript, Bootstrap
- **AI Models**: Support for OpenAI GPT models and Gemini models
- **Search Tools**: Integration with Tavily, Exa, and DuckDuckGo for web search

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd AI-chatroom
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your API keys:

Copy `env.example` to `.env` and fill in your API keys:

```bash
cp env.example .env
# Edit .env with your API keys
```

The application uses the following API keys:
- `OPENAI_API_KEY`: Required for OpenAI models (GPT-4o, GPT-4o-mini)
- `GEMINI_API_KEY`: Required for Google models (Gemini 2.5 Flash Preview, Gemini 1.5 Pro)
- `TAVILY_API_KEY`: For Tavily search (highest priority, optional)
- `EXA_API_KEY`: For Exa search (second priority, optional)

If no search API keys are provided, DuckDuckGo will be used as a fallback (no API key required).

## Supported Models

The application currently supports the following AI models:

### OpenAI Models
- GPT-4o
- GPT-4o Mini

### Gemini Models
- Gemini 2.5 Flash Preview
- Gemini 1.5 Pro

If you attempt to use a model without providing the corresponding API key, the system will automatically fall back to GPT-4o-mini.

## Usage

1. Start the application:

```bash
python main.py
```

2. Open your browser and navigate to `http://localhost:8000`

3. Create a new chatroom:
   - Enter a topic title
   - Add AI agents with names, models, and roles
   - Enable search capabilities for agents that need real-time information
   - Set the maximum number of discussion turns
   - Click "Create Discussion"

4. In the chatroom view:
   - Click "Start Discussion" to begin the AI conversation
   - Watch as the AI agents discuss the topic
   - Agents with search capabilities will use web search when needed
   - Search usage is indicated with a search icon next to the message
   - The discussion will automatically progress through the specified number of turns

## Search Tool Configuration

The application prioritizes search tools in the following order:

1. **Tavily Search**: Used first when `TAVILY_API_KEY` is provided
2. **Exa Search**: Used when `EXA_API_KEY` is provided
3. **DuckDuckGo**: Used as a fallback when no other search APIs are available

If a primary search tool fails (e.g., due to rate limiting or API issues), the system automatically falls back to DuckDuckGo.

Search tool usage is logged in the `logs/search_logs.json` file for monitoring and debugging purposes.

## Extending the Project

### Adding New AI Models

To add support for additional AI models:

1. Update the `create_model_from_string` function in `models.py` to include the new model in the mapping
2. Add the necessary API key to your `.env` file
3. Update the frontend model dropdown in `templates/index.html` to include the new model option

For example, to add a new model called "new-model" from a provider called "new-provider":

```python
# In models.py
model_mapping = {
    # Existing models...
    
    # New model
    "new-model": (ModelPlatformType.NEW_PROVIDER, ModelType.NEW_MODEL, NewProviderConfig().as_dict()),
}
```

### Adding New Search Tools

To add support for additional search tools:

1. Update the `create_search_tools` function in `models.py`
2. Add the necessary API keys to your `.env` file
3. Modify the prioritization logic as needed

### Customizing the UI

The UI templates are located in the `templates` directory and can be modified to change the appearance and functionality of the application.

## License

MIT