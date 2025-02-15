from __future__ import annotations

import deeplake.types

__all__ = [
    "TextEmbeddings",
    "COCOImages",
    "SchemaTemplate",
]

def TextEmbeddings(embedding_size: int, quantize: bool = False) -> dict[str, deeplake._deeplake.types.DataType | str | deeplake._deeplake.types.Type]:
    """
    A schema for storing embedded text from documents.

    This schema includes the following fields:
    - id (uint64): Unique identifier for each entry.
    - chunk_index (uint16): Position of the text chunk within the document.
    - document_id (uint64): Unique identifier for the document the embedding came from.
    - date_created (uint64): Timestamp when the document was read.
    - text_chunk (text): The text of the shard.
    - embedding (dtype=float32, size=embedding_size): The embedding of the text.

    <!-- test-context
    ```python
    import deeplake
    from deeplake import types
    ds = deeplake.create("tmp://")
    ```
    -->

    Parameters:
        embedding_size: int
            Size of the embeddings.
        quantize: bool, optional
            If true, quantize the embeddings to slightly decrease accuracy while greatly increasing query speed. Default is False.

    Examples:
        Create a dataset with the standard schema:
        ```python
        ds = deeplake.create("tmp://", schema=deeplake.schemas.TextEmbeddings(768))
        ```

        Customize the schema before creating the dataset:
        ```python
        schema = deeplake.schemas.TextEmbeddings(768)
        schema["text_embed"] = schema.pop("embedding")
        schema["author"] = types.Text()
        ds = deeplake.create("tmp://", schema=schema)
        ```

        Add a new field to the schema:
        ```python
        schema = deeplake.schemas.TextEmbeddings(768)
        schema["language"] = types.Text()
        ds = deeplake.create("tmp://", schema=schema)
        ```
    """
    ...

def COCOImages(
    embedding_size: int,
    quantize: bool = False,
    objects: bool = True,
    keypoints: bool = False,
    stuffs: bool = False,
) -> dict[str, deeplake._deeplake.types.DataType | str | deeplake._deeplake.types.Type]:
    """
    A schema for storing COCO-based image data.

    This schema includes the following fields:
    - id (uint64): Unique identifier for each entry.
    - image (jpg image): The image data.
    - url (text): URL of the image.
    - year (uint8): Year the image was captured.
    - version (text): Version of the dataset.
    - description (text): Description of the image.
    - contributor (text): Contributor of the image.
    - date_created (uint64): Timestamp when the image was created.
    - date_captured (uint64): Timestamp when the image was captured.
    - embedding (embedding): Embedding of the image.
    - license (text): License information.
    - is_crowd (bool): Whether the image contains a crowd.

    If `objects` is true, the following fields are added:
    - objects_bbox (bounding box): Bounding boxes for objects.
    - objects_classes (segment mask): Segment masks for objects.

    If `keypoints` is true, the following fields are added:
    - keypoints_bbox (bounding box): Bounding boxes for keypoints.
    - keypoints_classes (segment mask): Segment masks for keypoints.
    - keypoints (2-dimensional array of uint32): Keypoints data.
    - keypoints_skeleton (2-dimensional array of uint16): Skeleton data for keypoints.

    If `stuffs` is true, the following fields are added:
    - stuffs_bbox (bounding boxes): Bounding boxes for stuffs.
    - stuffs_classes (segment mask): Segment masks for stuffs.

    Parameters:
        embedding_size: int
            Size of the embeddings.
        quantize: bool, optional
            If true, quantize the embeddings to slightly decrease accuracy while greatly increasing query speed. Default is False.
        objects: bool, optional
            Whether to include object-related fields. Default is True.
        keypoints: bool, optional
            Whether to include keypoint-related fields. Default is False.
        stuffs: bool, optional
            Whether to include stuff-related fields. Default is False.

    Examples:
        Create a dataset with the standard schema:
        ```python
        ds = deeplake.create("tmp://", schema=deeplake.schemas.COCOImages(768))
        ```

        Customize the schema before creating the dataset:
        ```python
        schema = deeplake.schemas.COCOImages(768, objects=True, keypoints=True)
        schema["image_embed"] = schema.pop("embedding")
        schema["author"] = types.Text()
        ds = deeplake.create("tmp://", schema=schema)
        ```

        Add a new field to the schema:
        ```python
        schema = deeplake.schemas.COCOImages(768)
        schema["location"] = types.Text()
        ds = deeplake.create("tmp://", schema=schema)
        ```
    """
    ...
