from .common.model_schemas import ResearchToolOutput, ContentItem
from .base_tool import ResearchTool

from importlib import import_module

_MODULES = {
    "ExaCompanySearch": ".exa_company_search",
    "NewsSearch": ".news_search",
    "SimilarWebSearch": ".similar_web_search",
    "YouComSearch": ".you_com_search",
    "ParallelSearch": ".parallel_search",
    "ParallelFetch": ".parallel_search",
    "create_default_tools": ".defaults",
}
__all__ = list(_MODULES)


def __getattr__(name):
    # Legacy tools register with Eezo. Load them only when requested.
    if name not in _MODULES:
        raise AttributeError(name)
    return getattr(import_module(_MODULES[name], __name__), name)
