import logging
import dotenv
import os

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
research_agent = ResearchAgent(tools)


@e.on(os.environ["AGENT_RESEARCH"])
def research_agent_handler(context, **kwargs):
    research_agent.invoke(context, **kwargs)


e.connect()
