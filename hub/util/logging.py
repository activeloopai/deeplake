from hub.client.log import logger
from hub.util.path import is_hub_cloud_path, get_org_id_and_ds_name


def log_visualizer_link(org_id, ds_name, source_ds_url=""):
    msg = "This dataset can be visualized in Jupyter Notebook by ds.visualize()"
    url = f"https://app.activeloop.ai/{org_id}/{ds_name}"
    if url.endswith("/queries"):  # Ignore user queries ds
        pass
    elif "/.queries/" in url:  # Is a view
        if "/queries/" in url:  # Stored in user queries ds
            if is_hub_cloud_path(source_ds_url):
                org_id, ds_name = get_org_id_and_ds_name(source_ds_url)
                source_ds_url = f"https://app.activeloop.ai/{org_id}/{ds_name}"
                view_id = url.split("/.queries/", 1)[1]
                if view_id.endswith("_OPTIMIZED"):
                    view_id = view_id[: -len("_OPTIMIZED")]
                url = source_ds_url + "?view=" + view_id
                logger.info(msg + " or at " + url)
            else:
                logger.info(msg + ".")
        else:  # Stored in ds
            ds_url, view_id = url.split("/.queries/", 1)
            if view_id.endswith("_OPTIMIZED"):
                view_id = view_id[: -len("_OPTIMIZED")]
            url = ds_url + "?view=" + view_id
            logger.info(msg + " or at " + url)
    else:
        logger.info(msg + " or at " + url)
