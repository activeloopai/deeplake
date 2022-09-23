import sys


def is_notebook():
    """Whether running in a notebook."""
    try:
        from IPython import get_ipython  # type: ignore

        if get_ipython() is None:
            return False
    except ImportError:
        return False
    return True


def is_jupyter():
    """Whether running in a Jupyter notebook."""
    if not is_notebook():
        return False
    from IPython import get_ipython

    if "terminal" in get_ipython().__module__ or "spyder" in sys.modules:
        return False
    return True


def is_colab():
    """Whether running in a colab notebook."""
    return "google.colab" in sys.modules


def video_html(src, alt):
    import IPython  # type: ignore

    html = f"""<video alt="{alt}" width=500 controls autoplay seek loop>
                    <source src="{src}" type="video/mp4">
                </video>
            """
    return IPython.display.HTML(html)
