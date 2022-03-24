from IPython.display import IFrame

from hub.core.dataset.dataset import Dataset

class VisualContext:
    """
    VisualContext class is the main class to interact with the visualizer.
    Users should not create the instance of this class directly, instead they will
    obtain the context from the visualize() function. The workflow for interacting 
    with the visualizer will be the following:
    >>> ds = hub.load(...)
    >>> context = visualize(ds)
    >>> with context:
    >>>     #Do changes on the dataset
    >>>
    >>> # Visualizer will be updated automatically.

    For the cases when force update is needed, the update() function can be used.
    """
    _id: str = ""
    _iframe: IFrame = None
    _ds: Dataset = None

    def __init__(self, id: str, ds: Dataset, iframe: IFrame):
        self._id = id
        self._iframe = iframe
        self._ds = ds

    @property
    def id(self):
        return self._id

    @property
    def ds(self):
        return self._ds

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def update():
        pass