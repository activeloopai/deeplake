from .api.dataset import Dataset
from .util.bugout_reporter import hub_reporter, hub_tags

__version__ = "2.0.0"

# Reporting
hub_tags.extend(__version__)
hub_reporter.system_report(publish=True, tags=hub_tags)
hub_reporter.setup_excepthook(publish=True, tags=hub_tags)
