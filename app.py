import logging
import dotenv

# !Rename env to .env and add missing keys!
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

from research_agent import ResearchAgent
from eezo import Eezo
from tools import *


e = Eezo()

# To connect more sources, copy one of the existing tools in the tools/research
# folder and connect it to your data source. Then, add it to the list below.
tools = [YouComSearch(), SimilarWebSearch(), ExaCompanySearch(), NewsSearch()]

# Check if the agent already exists, if not, create it.
ra = e.get_agent("research_agent")
if ra is None:
    e.create_agent(
        agent_id="research_agent",
        description="Invoke when the user wants to perform a research task.",
    )

# Create an instance of the ResearchAgent class and pass the tools list to it.
research_agent = ResearchAgent(tools)


# Define the handler for the research_agent event.
@e.on("research_agent")
def research_agent_handler(context, **kwargs):
    research_agent.invoke(context, **kwargs)


e.connect()
