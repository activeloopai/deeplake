import os

from typing import List, Optional, Sequence, Union

# TwelveLabs Marengo produces 512-dimensional multimodal embeddings.
MARENGO_EMBEDDING_DIM = 512

# Default Marengo model name. See https://docs.twelvelabs.io for available models.
DEFAULT_MARENGO_MODEL = "marengo3.0"


class MarengoEmbedder:
    """Generates multimodal embeddings with TwelveLabs Marengo for Deep Lake datasets.

    Marengo maps text, images, and audio into the **same** 512-dimensional vector
    space, which makes it a good fit for building multimodal vector stores in Deep
    Lake: embed your media at ingestion time and embed natural-language queries at
    search time, then run cosine-similarity search against a single
    :func:`deeplake.types.Embedding` column.

    This integration is fully opt-in. It does not change any Deep Lake defaults and
    requires the optional ``twelvelabs`` package (``pip install twelvelabs``) plus a
    TwelveLabs API key. You can grab a free API key at https://twelvelabs.io.

    Args:
        api_key (str, optional): TwelveLabs API key. If omitted, the ``TWELVELABS_API_KEY``
            environment variable is used.
        model_name (str, optional): Marengo model to use. Defaults to ``"marengo3.0"``.

    Raises:
        ValueError: If no API key is provided and ``TWELVELABS_API_KEY`` is not set.

    Example:
        >>> import deeplake
        >>> from deeplake import types
        >>> from deeplake.integrations.twelvelabs import MarengoEmbedder
        >>>
        >>> embedder = MarengoEmbedder(api_key="<your_api_key>")
        >>>
        >>> ds = deeplake.create("mem://videos")
        >>> ds.add_column("embedding", types.Embedding(embedder.dim))
        >>> ds.add_column("url", types.Text())
        >>>
        >>> # Embed videos at ingestion time.
        >>> video_urls = ["https://example.com/clip.mp4"]
        >>> ds.append({
        ...     "url": video_urls,
        ...     "embedding": embedder.embed_videos(video_urls),
        ... })
        >>> ds.commit()
        >>>
        >>> # Embed a text query at search time and run vector search.
        >>> query = ",".join(str(c) for c in embedder.embed_text("a chef plating a dish"))
        >>> view = ds.query(
        ...     f"SELECT * ORDER BY COSINE_SIMILARITY(embedding, ARRAY[{query}]) DESC LIMIT 5"
        ... )

    Notes:
        - All embeddings are 512-dimensional vectors in a shared multimodal space,
          suitable for cosine-similarity search.
        - ``embed_text`` follows the same calling convention as the
          ``embedding_function`` used throughout the Deep Lake RAG guides
          (string or list of strings in, list of vectors out), so it can be dropped
          into existing pipelines.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MARENGO_MODEL,
    ):
        api_key = api_key or os.environ.get("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "A TwelveLabs API key is required. Pass api_key=... or set the "
                "TWELVELABS_API_KEY environment variable. Get a free key at "
                "https://twelvelabs.io."
            )

        from twelvelabs import TwelveLabs  # type: ignore

        self._client = TwelveLabs(api_key=api_key)
        self.model_name = model_name

    @property
    def dim(self) -> int:
        """Embedding dimensionality (512), for use with ``types.Embedding(embedder.dim)``."""
        return MARENGO_EMBEDDING_DIM

    @staticmethod
    def _first_segment_vector(embedding) -> List[float]:
        if embedding is None or not embedding.segments:
            raise ValueError(
                "TwelveLabs returned no embedding segments. "
                f"error_message={getattr(embedding, 'error_message', None)}"
            )
        return list(embedding.segments[0].float_)

    def embed_text(self, texts: Union[str, Sequence[str]]) -> List[List[float]]:
        """Embed one or more text strings into 512-dimensional vectors.

        Args:
            texts (str | Sequence[str]): A single string or a sequence of strings.

        Returns:
            List[List[float]]: One 512-dimensional vector per input string.
        """
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            response = self._client.embed.create(model_name=self.model_name, text=text)
            vectors.append(self._first_segment_vector(response.text_embedding))
        return vectors

    def embed_videos(self, video_urls: Union[str, Sequence[str]]) -> List[List[float]]:
        """Embed one or more videos (by public URL) into 512-dimensional vectors.

        Marengo segments a video and returns one embedding per segment; this method
        returns the first segment's vector for each video so the output lines up
        one-to-one with the input, ready to append to an ``Embedding`` column. For
        per-segment access, call the TwelveLabs SDK directly.

        Args:
            video_urls (str | Sequence[str]): A single video URL or a sequence of URLs.

        Returns:
            List[List[float]]: One 512-dimensional vector per input video.
        """
        if isinstance(video_urls, str):
            video_urls = [video_urls]

        vectors = []
        for url in video_urls:
            task = self._client.embed.tasks.create(
                model_name=self.model_name, video_url=url
            )
            self._client.embed.tasks.wait_for_done(task_id=task.id)
            result = self._client.embed.tasks.retrieve(
                task_id=task.id, embedding_option=["visual-text"]
            )
            vectors.append(self._first_segment_vector(result.video_embedding))
        return vectors
