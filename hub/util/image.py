from io import BytesIO
from PIL import Image  # type: ignore
from hub.core.sample import Sample


def convert_sample(image_sample: Sample, mode: str, compression: str) -> Sample:
    if image_sample.path:
        image = Image.open(image_sample.path)
    elif image_sample._buffer:
        image = Image.open(BytesIO(image_sample._buffer))
    if image.mode == mode:
        return image_sample

    image = image.convert(mode)
    image_bytes = BytesIO()
    image.save(image_bytes, format=compression)
    converted = Sample(
        buffer=image_bytes.getvalue(), compression=image_sample.compression
    )
    return converted
