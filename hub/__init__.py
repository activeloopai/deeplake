from .api.dataset import Dataset
from .util.bugout_reporter import reporter

__version__ = "2.0.0"

# Reporting
reporter.system_report(publish=True, tags=[__version__])
reporter.setup_excepthook(publish=True, tags=[__version__])
