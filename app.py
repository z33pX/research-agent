import logging
import dotenv

# !Rename env to .env and add missing keys!
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

from research_agent import ResearchAgent
from eezo.agent import Agent
from eezo import Eezo
from tools import *


e = Eezo()

# To connect more sources, copy one of the existing tools in the tools/research
# folder and connect it to your data source. Then, add it to the list below.
tools = [YouComSearch(), SimilarWebSearch(), ExaCompanySearch(), NewsSearch()]

# Check if the agent already exists, if not, create it.
ra: Agent = e.get_agent("research-agent")
if ra is None:
    e.create_agent(
        agent_id="research-agent",
        description="Invoke when the user wants to perform a research task.",
    )

# Create an instance of the ResearchAgent class and pass the tools list to it.
research_agent = ResearchAgent(tools)


# Define the handler for the research_agent event.
@e.on("research-agent")
def research_agent_handler(context, **kwargs):
    research_agent.invoke(context, **kwargs)


# Define the handlers for the tools.
# We can use the same handler for all tools since they all have the same structure.


@e.on("you-com-search")
def you_com_search_handler(context, **kwargs):
    result = YouComSearch(include_summary=True).invoke(input=kwargs)
    m = context.new_message()
    m.add("text", text=result.summary)
    m.notify()


@e.on("similar-web-search")
def similar_web_search_handler(context, **kwargs):
    result = SimilarWebSearch(include_summary=True).invoke(input=kwargs)
    m = context.new_message()
    m.add("text", text=result.summary)
    m.notify()


@e.on("exa-company-search")
def exa_company_search_handler(context, **kwargs):
    result = ExaCompanySearch(include_summary=True).invoke(input=kwargs)
    m = context.new_message()
    m.add("text", text=result.summary)
    m.notify()


@e.on("news-search")
def news_search_handler(context, **kwargs):
    result = NewsSearch(include_summary=True).invoke(input=kwargs)
    m = context.new_message()
    m.add("text", text=result.summary)
    m.notify()


e.connect()
