from __future__ import annotations

import deeplake.types

__all__ = [
    "TextEmbeddings",
    "COCOImages",
    "SchemaTemplate",
]

def TextEmbeddings(embedding_size: int, quantize: bool = False) -> SchemaTemplate:
    """
    A schema for storing embedded text from documents.

    - id (uint64)
    - chunk_index (uint16) Position of the text_chunk within the document
    - document_id (uint64) Unique identifier for the document the embedding came from
    - date_created (uint64) Timestamp the document was read
    - text_chunk (text) The text of the shard
    - embedding (dtype=float32, size=embedding_size) The embedding of the text

    Parameters:
         embedding_size: Size of the embeddings
         quantize: If true, quantize the embeddings to slightly decrease accuracy while greatly increasing query speed

    Examples:
        ```python
        # Create a dataset with the standard schema
        ds = deeplake.create("ds_path",
                schema=deeplake.schemas.TextEmbeddings(768).build())

        # Customize the schema before creating the dataset
        ds = deeplake.create("ds_path", schema=deeplake.schemas.TextEmbeddings(768)
                .rename("embedding", "text_embed")
                .add("author", types.Text())
                .build())
        ```

    """
    ...

def COCOImages(
    embedding_size: int,
    quantize: bool = False,
    objects: bool = True,
    keypoints: bool = False,
    stuffs: bool = False,
) -> SchemaTemplate:
    """
    A schema for storing COCO-based image data.

        - id (uint64)
        - image (jpg image)
        - url (text)
        - year (uint8)
        - version (text)
        - description (text)
        - contributor (text)
        - date_created (uint64)
        - date_captured (uint64)
        - embedding (embedding)
        - license (text)
        - is_crowd (bool)

    If `objects` is true, the following fields are added:
        - objects_bbox (bounding box)
        - objects_classes (segment mask)

    If `keypoints` is true, the following fields are added:
        - keypoints_bbox (bounding box)
        - keypoints_classes (segment mask)
        - keypoints (2-dimensional array of uint32)
        - keypoints_skeleton (2-dimensional array of uint16)

    if `stuffs` is true, the following fields are added:
        - stuffs_bbox (bounding boxes)
        - stuffs_classes (segment mask)

    Parameters:
         embedding_size: Size of the embeddings
         quantize: If true, quantize the embeddings to slightly decrease accuracy while greatly increasing query speed

    Examples:
        ```python
        # Create a dataset with the standard schema
        ds = deeplake.create("ds_path",
            schema=deeplake.schemas.COCOImages(768).build())

        # Customize the schema before creating the dataset
        ds = deeplake.create("ds_path", schema=deeplake.schemas.COCOImages(768,
                objects=True, keypoints=True)
            .rename("embedding", "image_embed")
            .add("author", types.Text()).build())
        ```

    """
    ...

class SchemaTemplate:
    """
    A template that can be used for creating a new dataset with [deeplake.create][]
    """

    # Temporary workaround. Need to remove `deeplake._deeplake` from the return type.
    def __init__(
        self,
        schema: dict[
            str, deeplake._deeplake.types.DataType | str | deeplake._deeplake.types.Type
        ],
    ) -> None:
        """
        Constructs a new SchemaTemplate from the given dict
        """
        ...

    def add(
        self,
        name: str,
        dtype: deeplake._deeplake.types.DataType | str | deeplake._deeplake.types.Type,
    ) -> SchemaTemplate:
        """
        Adds a new column to the template

        Parameters:
            name: The column name
            dtype: The column data type
        """
        ...

    def remove(self, name: str) -> SchemaTemplate:
        """
        Removes a column from the template

        Parameters:
            name: The column name
        """
        ...

    def rename(self, old_name: str, new_name: str) -> SchemaTemplate:
        """
        Renames a column in the template.

        Parameters:
            old_name: Existing column name
            new_name: New column name
        """
        ...
