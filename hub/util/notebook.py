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

    html = f"""<video alt="{alt}" width=500 controls autoplay seek loop>
                    <source src="{src}" type="video/mp4">
                </video>
            """
    with open("video.html", "w") as f:
        f.write(html)
    return IPython.display.HTML(filename=src)
