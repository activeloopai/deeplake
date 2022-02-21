import sys


def is_notebook():
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
    except ImportError:
        return False
    return True


def is_jupyter():
    if not is_notebook():
        return False
    from IPython import get_ipython

    if "terminal" in get_ipython().__module__ or "spyder" in sys.modules:
        return False
    return True


def is_colab():
    return "google.colab" in sys.modules


def video_html(src, alt):
    import IPython

    html = f"""<video alt="{alt}" width=500 controls autoplay seek loop allowfullscreen>
                    <source src="{src}" type="video/mp4">
                </video>
            """
    html = """<iframe width="560" height="315" src="{src}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
    """
    return IPython.display.HTML(html)
