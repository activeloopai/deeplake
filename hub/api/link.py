from hub.core.linked_sample import LinkedSample
from typing import Optional, Dict


def link(
    path: str,
    creds_key: Optional[str] = None,
) -> LinkedSample:
    """Utility that stores a link to raw data. Used to add data to a Hub Dataset without copying it.

    Note:
        No data is actually loaded until you try to read the sample from a dataset.
        There are a few exceptions to this:-

        - If verify=True was specified DURING create_tensor of the tensor to which this is being added, some metadata is read to verify the integrity of the sample.
        - If create_shape_tensor=True was specified DURING create_tensor of the tensor to which this is being added, the shape of the sample is read.
        - If create_sample_info_tensor=True was specified DURING create_tensor of the tensor to which this is being added, the sample info is read.

    Examples:
        >>> ds = hub.dataset("......")

        Add the names of the creds you want to use (not needed for http/local urls)
        >>> ds.add_creds("MY_S3_KEY")
        >>> ds.add_creds("GCS_KEY")

        Populate the names added with creds dictionary
        These creds are only present temporarily and will have to be repopulated on every reload
        >>> ds.populate_creds("MY_S3_KEY", {})
        >>> ds.populate_creds("GCS_KEY", {})

        Create a tensor that can contain links
        >>> ds.create_tensor("img", htype="link[image]", verify=True, create_shape_tensor=False, create_sample_info_tensor=False)

        Populate the tensor with links
        >>> ds.img.append(hub.link("s3://abc/def.jpeg", creds_key="MY_S3_KEY"))
        >>> ds.img.append(hub.link("gcs://ghi/jkl.png", creds_key="GCS_KEY"))
        >>> ds.img.append(hub.link("https://picsum.photos/200/300")) # doesn't need creds
        >>> ds.img.append(hub.link("s3://abc/def.jpeg"))  # will use creds from environment
        >>> ds.img.append(hub.link("s3://abc/def.jpeg", creds_key="ENV"))  # this will also use creds from environment

        Accessing the data
        >>> for i in range(5):
        >>> ds.img[i].numpy()

        Updating a sample
        >>> ds.img[0] = hub.link("./data/cat.jpeg")

    Supported file types:

        Image: "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", "webp", "wmf", "xbm"
        Audio: "flac", "mp3", "wav"
        Video: "mp4", "mkv", "avi"
        Dicom: "dcm"

    Args:
        path (str): Path to a supported file.
        creds_key (optional, str): The credential key to use to read data for this sample. The actual credentials are fetched from the dataset.

    Returns:
        LinkedSample: LinkedSample object that stores path and creds.
    """
    return LinkedSample(path, creds_key)
