import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

from camel.models import ModelFactory
from camel.toolkits import *
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

from typing import List, Dict

from retry import retry
from loguru import logger

# Assuming OwlRolePlaying and run_society are from utils or defined elsewhere
from utils import OwlRolePlaying, run_society


def construct_society(question: str) -> OwlRolePlaying:
    """Construct the society based on the question."""

    user_role_name = "user"
    assistant_role_name = "assistant"
    
    # Create user model
    user_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),  # [Optional] the config for model
    )

    # Create assistant model
    assistant_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),  # [Optional] the config for model
    )

    # Define tools to be used in the society
    tools_list = [
        *WebToolkit(
            headless=False,  # Set to True if you want to run in headless mode (e.g. on a remote server)
            web_agent_model=assistant_model, 
            planning_agent_model=assistant_model
        ).get_tools(),
        *DocumentProcessingToolkit().get_tools(),
        *VideoAnalysisToolkit(model=assistant_model).get_tools(),  # This requires OpenAI Key
        *AudioAnalysisToolkit().get_tools(),  # This requires OpenAI Key
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=assistant_model).get_tools(),
        *SearchToolkit(model=assistant_model).get_tools(),
        *ExcelToolkit().get_tools()
    ]

    # Set agent details
    user_role_name = 'user'
    user_agent_kwargs = dict(model=user_model)
    assistant_role_name = 'assistant'
    assistant_agent_kwargs = dict(model=assistant_model, tools=tools_list)

    task_kwargs = {
        'task_prompt': question,
        'with_task_specify': False,
    }

    # Construct society instance
    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name=user_role_name,
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name=assistant_role_name,
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    
    return society


# Example case
question = "What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper 'Can Hiccup Supply Enough Fish to Maintain a Dragon's Diet?'"

# Construct society with the question
society = construct_society(question)

# Run the society and capture the answer
answer, chat_history, token_count = run_society(society)

# Log the result
logger.success(f"Answer: {answer}")
