import search_tools
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions about the repository.  

Always use the search tool to find relevant information before answering.  

If no relevant results are found, say so clearly.

Always include references by citing the filename of the source material you used.
Replace it with the full path to the GitHub repository:
"https://github.com/{repo_owner}/{repo_name}/tree/main"
Format: [LINK TITLE](FULL_GITHUB_LINK)

If the search doesn't return relevant results, let the user know and provide general guidance.""".strip()

def init_agent(pytorch_img_index, pytorch_img_vindex, repo_owner, repo_name):

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(repo_owner=repo_owner, repo_name=repo_name)

    search_tool = search_tools.SearchTool(pytorch_img_index=pytorch_img_index, pytorch_img_vindex=pytorch_img_vindex)

    # agent = Agent(
    # name="pytorch_img_agent",
    # instructions=system_prompt,
    # tools=[search_tool.search],
    # model=GroqModel(
    #     'llama-3.3-70b-versatile',
    #     provider=GroqProvider(api_key=os.getenv('GROQ_API_KEY'))
    #     )
    # )

    agent = Agent(
    name="pytorch_img_agent",
    instructions=system_prompt,
    tools=[search_tool.search],
    model=OpenAIChatModel(
        'gpt-4o-mini',
        provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))
        )
    )    

    return agent