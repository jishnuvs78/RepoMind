import search_tools
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

SYSTEM_PROMPT_TEMPLATE = """
You are an assistant for the PyTorch Image Models (timm) repository.

Always use the search tools to find relevant information before answering.

- Use hybrid_search_repo for the queries

If no relevant results are found, say so clearly.

IMPORTANT:
When calling a tool, return ONLY a valid tool call in JSON format.
Do not output text like <function=...>.

Always include references by citing the filename of the source material you used.  
When citing the reference, provide the full path to the GitHub repository: "https://github.com/{repo_owner}/{repo_name}/tree/main"
Format: [LINK TITLE](FULL_GITHUB_LINK)

If the search doesn't return relevant results, let the user know and provide general guidance.
""".strip()

def init_agent(pytorch_img_index, pytorch_img_vindex, repo_owner, repo_name):

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(repo_owner=repo_owner, repo_name=repo_name)

    search_tool = search_tools.SearchTool(pytorch_img_index=pytorch_img_index, pytorch_img_vindex=pytorch_img_vindex)

    agent = Agent(
    name="pytorch_img_agent",
    instructions=system_prompt,
    tools=[search_tool.search],
    model=GroqModel('llama-3.3-70b-versatile')
    )

    return agent