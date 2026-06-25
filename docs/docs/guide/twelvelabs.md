---
seo_title: "TwelveLabs Marengo embeddings for Deep Lake | Multimodal video vector search"
description: "Build a multimodal video vector store in Deep Lake using TwelveLabs Marengo 512-dim embeddings for text and video."
---

# Video Vector Search with TwelveLabs Marengo

[TwelveLabs](https://twelvelabs.io) Marengo is a multimodal embedding model that maps
text, images, audio, and **video** into the same 512-dimensional vector space. Because
Deep Lake stores raw video alongside embeddings and runs vector search client-side, the
two pair naturally: embed your videos at ingestion time, embed natural-language queries
at search time, and run cosine-similarity search over a single `Embedding` column.

This integration is fully opt-in and does not change any Deep Lake defaults.

## Install

```bash
pip install deeplake twelvelabs
```

Set your TwelveLabs API key (grab a free one at [twelvelabs.io](https://twelvelabs.io) —
there's a generous free tier):

```python
import os
os.environ["TWELVELABS_API_KEY"] = "<your_api_key>"
```

## Create a dataset and embed videos

`MarengoEmbedder.dim` is `512`, so it plugs straight into `types.Embedding`. The
`embed_videos` method accepts public video URLs and returns one vector per video.

```python
import deeplake
from deeplake import types
from deeplake.integrations.twelvelabs import MarengoEmbedder

embedder = MarengoEmbedder()  # reads TWELVELABS_API_KEY from the environment

ds = deeplake.create("mem://videos")
ds.add_column("url", types.Text())
ds.add_column("embedding", types.Embedding(embedder.dim))

video_urls = [
    "https://example.com/cooking.mp4",
    "https://example.com/traffic.mp4",
]
ds.append({
    "url": video_urls,
    "embedding": embedder.embed_videos(video_urls),
})
ds.commit()
```

## Search with a text query

Because Marengo embeds text into the same space as video, you can search your videos
with natural language. `embed_text` follows the same calling convention as the
`embedding_function` used throughout the Deep Lake RAG guides (string or list of strings
in, list of vectors out).

```python
query_vector = embedder.embed_text("a chef plating a dish in a busy kitchen")[0]
query_str = ",".join(str(c) for c in query_vector)

view = ds.query(
    f"SELECT *, COSINE_SIMILARITY(embedding, ARRAY[{query_str}]) AS score "
    f"ORDER BY score DESC LIMIT 5"
)
```

`embed_text` also accepts a list of strings and returns the vectors in order, so it can
be dropped into existing batched ingestion pipelines.
