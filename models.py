import os
from dotenv import load_dotenv
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelType, ModelPlatformType
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig

try:
    from camel.configs import GeminiConfig
except ImportError:
    GeminiConfig = None

load_dotenv()

def create_camel_agent(model_name_str: str, role_description: str):
    try:
        model_obj = None
        if model_name_str == "gpt-4o-mini":
            model_obj = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict=ChatGPTConfig().as_dict()
            )
        elif model_name_str == "gemini-1.5-flash":
            config_dict = {}
            if GeminiConfig:
                config_dict = GeminiConfig().as_dict()

            model_obj = ModelFactory.create(
                model_platform=ModelPlatformType.GEMINI,
                model_type="gemini-1.5-flash",
                model_config_dict=config_dict,
            )
        else:
            raise ValueError(f"Model '{model_name_str}' is not explicitly supported by this factory function.")

        if model_obj is None:
             raise ValueError(f"Failed to create model instance for {model_name_str} using ModelFactory.")

        print(f"Successfully created model instance for: {model_name_str} with role: {role_description}")
        return model_obj

    except ImportError as ie:
        print(f"ImportError during agent creation for {model_name_str}: {ie}.")
        raise ValueError(f"Failed to create agent for model {model_name_str} due to missing import: {str(ie)}")
    except Exception as e:
        print(f"Error creating ChatAgent for model {model_name_str} with role {role_description}: {e}")
        raise ValueError(f"Failed to create agent for model {model_name_str}: {str(e)}")
