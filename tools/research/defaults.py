"""Select the app's general web tools without changing explicit tool lists."""
import os
from uuid import uuid4


def create_default_tools(provider=None):
    """Explicit argument overrides RESEARCH_WEB_SEARCH; none disables general web tools."""
    choice = provider if provider is not None else os.getenv("RESEARCH_WEB_SEARCH", "auto")
    choice = choice.strip().lower()
    if choice not in {"auto", "parallel", "youcom", "none"}:
        raise ValueError("RESEARCH_WEB_SEARCH must be auto, parallel, youcom, or none")
    if choice == "auto":
        choice = "youcom" if os.getenv("YOUCOM_API_KEY", "").strip() else "parallel"

    from . import ExaCompanySearch, NewsSearch, SimilarWebSearch

    general = []
    if choice == "youcom":
        from . import YouComSearch
        general = [YouComSearch()]
    elif choice == "parallel":
        from .parallel_search import ParallelSearch, ParallelFetch
        session_id = str(uuid4())
        general = [ParallelSearch(session_id=session_id), ParallelFetch(session_id=session_id)]
    return general + [SimilarWebSearch(), ExaCompanySearch(), NewsSearch()]
