# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import os
import time
import random
import traceback

# Fix Unicode escape errors with paths
from dotenv import load_dotenv

# Use raw strings or forward slashes for paths
print("Current working directory:", os.getcwd())
env_path = os.path.join(os.getcwd(), '.env')
print(f"Looking for .env file at: {env_path}")

# Use a try-except block to handle errors
try:
    load_dotenv(dotenv_path=env_path)
    print("Successfully loaded .env file")
except Exception as e:
    print(f"Error loading .env file: {str(e)}")
    
    # Try loading without explicit path
    try:
        load_dotenv()
        print("Successfully loaded .env without explicit path")
    except Exception as e:
        print(f"Error loading .env without path: {str(e)}")

# Debug environment variables
print("Environment variables:")
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY", "Not found"))
print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY", "Not found"))

# Set up OpenRouter with the OPENAI_API_KEY if OPENROUTER_API_KEY is not found
openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openrouter_key:
    # If no keys found, ask for key directly
    print("No API key found in environment variables.")
    openrouter_key = input("Please enter your OpenRouter API key: ")
    os.environ["OPENAI_API_KEY"] = openrouter_key  # Set it for this session

# Set environment variables for CAMEL
os.environ["OPENAI_API_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = openrouter_key

# Now import the rest of the modules
from openai import OpenAI
from camel.models import ModelFactory
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    WebToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from retry import retry
from loguru import logger
from utils import OwlRolePlaying, run_society, DocumentProcessingToolkit

# Create OpenRouter client
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "OWL Framework"
    }
)

def safe_sleep(base_seconds=2):
    """Sleep with jitter to avoid predictable patterns"""
    jitter = random.uniform(0.5, 1.5)
    sleep_time = base_seconds * jitter
    print(f"Sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)

# Test OpenRouter connection
def test_openrouter():
    try:
        print("Testing OpenRouter connection...")
        if openrouter_key and len(openrouter_key) > 5:
            print(f"Using API key (first 5 chars): {openrouter_key[:5]}...")
        else:
            print("Warning: API key seems too short or empty")
            
        completion = openai_client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'OpenRouter is working!'"
                }
            ]
        )
        print(f"OpenRouter test response: {completion.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"OpenRouter test failed: {str(e)}")
        return False

def construct_society(question: str) -> OwlRolePlaying:
    """Construct the society with OpenRouter configuration."""
    
    print("Creating user model with OpenRouter...")
    # Create user model with OpenRouter
    user_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
    )
    
    safe_sleep(2)
    
    print("Creating assistant model with OpenRouter...")
    assistant_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
    )
    
    print("Setting up WebToolkit...")
    tools_list = [
        *WebToolkit(
            headless=False,
            web_agent_model=assistant_model, 
            planning_agent_model=assistant_model
        ).get_tools()
    ]
    
    return OwlRolePlaying(
        task_prompt=question,
        with_task_specify=False,
        user_role_name='user',
        user_agent_kwargs=dict(model=user_model),
        assistant_role_name='assistant',
        assistant_agent_kwargs=dict(
            model=assistant_model,
            tools=tools_list
        )
    )

def run_with_retries(society):
    """Run society with retries and backoff for OpenRouter."""
    max_retries = 3
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries} to run society...")
            result = run_society(society)
            print("Run completed successfully!")
            return result
        except Exception as e:
            print(f"Error during run (attempt {attempt+1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                print("Max retries exceeded. Last error: " + str(e))
                raise Exception("Max retries exceeded: " + str(e))

def main():
    try:
        # Test OpenRouter connection first
        if not test_openrouter():
            print("Failed to connect to OpenRouter. Please check your API key and configuration.")
            return
            
        # Simple task for testing
        question = "find best ai tools in the market"
        
        print("\n" + "="*50)
        print("Starting with OpenRouter configuration...")
        if openrouter_key and len(openrouter_key) > 5:
            print(f"Using OpenRouter with API key (first 5 chars): {openrouter_key[:5]}...")
        print("API Base URL: https://openrouter.ai/api/v1")
        print("="*50 + "\n")
        
        society = construct_society(question)
        print("Society constructed successfully!")
        
        print("\nRunning task...")
        answer, chat_history, token_count = run_with_retries(society)
        
        print("\n" + "="*50)
        print("TASK EXECUTION COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Answer: {answer}")
        print(f"Token count: {token_count}")
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"CRITICAL ERROR: {str(e)}")
        print("="*50)
        traceback.print_exc()

if __name__ == "__main__":
    main()