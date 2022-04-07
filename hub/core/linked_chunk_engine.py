from typing import List, Optional, Union, Dict, Any

import numpy as np
from hub.compression import VIDEO_COMPRESSIONS
from hub.core.chunk.base_chunk import BaseChunk
from hub.core.chunk_engine import ChunkEngine
from hub.core.index.index import Index
from hub.core.link_creds import LinkCreds
from hub.core.linked_sample import LinkedSample
from hub.core.meta.encode.creds import CredsEncoder
from hub.core.storage import LRUCache
from hub.util.casting import get_dtype, get_htype
from hub.util.chunk_engine import check_sample_shape, check_samples_type
import hub
from hub.util.exceptions import ReadOnlyModeError
from hub.util.keys import get_creds_encoder_key


class LinkedChunkEngine(ChunkEngine):
    def __init__(
        self,
        key: str,
        cache: LRUCache,
        version_state: Dict[str, Any],
        meta_cache: LRUCache = None,
        link_creds: Optional[LinkCreds] = None,
    ):
        super().__init__(key, cache, version_state, meta_cache)
        self.link_creds = link_creds
        self._creds_encoder: Optional[CredsEncoder] = None
        self._creds_encoder_commit_id: Optional[str] = None

    @property
    def creds_encoder(self):
        commit_id = self.commit_id
        if self._creds_encoder is None or self._creds_encoder_commit_id != commit_id:
            commit_id = self.commit_id
            key = get_creds_encoder_key(self.key, commit_id)
            if not self.creds_encoder_exists:
                enc = CredsEncoder()
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_hub_object(key, CredsEncoder)
            self._creds_encoder = enc
            self._creds_encoder_commit_id = commit_id
            self.meta_cache.register_hub_object(key, enc)
        return self._creds_encoder

    @property
    def creds_encoder_exists(self):
        commit_id = self.commit_id
        if (
            self._creds_encoder is not None
            and self._creds_encoder_commit_id == commit_id
        ):
            return True
        try:
            key = get_creds_encoder_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    def numpy(
        self, index: Index, aslist: bool = False, use_data_cache: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        length = self.num_samples
        last_shape = None

        samples = []
        creds_encoder = self.creds_encoder

        if self.compression in VIDEO_COMPRESSIONS:
            raise NotImplementedError

        for global_sample_index in index.values[0].indices(length):
            sample_path = super().numpy(
                Index(global_sample_index), use_data_cache=False
            )[0]
            sample_creds_encoded = creds_encoder.get_encoded_creds_key(
                global_sample_index
            )
            sample_creds_key = self.link_creds.get_creds_key(sample_creds_encoded)
            if (
                sample_path.startswith("gcs://")
                or sample_path.startswith("gcp://")
                or sample_path.startswith("s3://")
            ):
                provider_type = "s3" if sample_path.startswith("s3://") else "gcs"
                storage = self.link_creds.get_storage_provider(
                    sample_creds_key, provider_type
                )
                sample = hub.read(sample_path, storage=storage)
            else:
                sample = hub.read(sample_path)
            sample = sample.array[tuple(entry.value for entry in index.values[1:])]
            samples.append(sample)
            check_sample_shape(sample.shape, last_shape, self.key, index, aslist)
            last_shape = sample.shape

        if aslist and all(map(np.isscalar, samples)):
            samples = [arr.item() for arr in samples]

        if not index.values[0].subscriptable():
            samples = samples[0]

        if aslist:
            return samples
        return np.array(samples)

    # TODO, override def extend for callbacks

    def check_each_sample(self, samples):
        for sample in samples:
            if not isinstance(sample, LinkedSample):
                raise TypeError(
                    f"Expected LinkedSample, got {type(sample)} instead. Use hub.link() to link samples."
                )

    def _samples_to_chunks(
        self,
        samples,
        start_chunk: Optional[BaseChunk] = None,
        register: bool = True,
        update_commit_diff: bool = False,
    ):
        creds = [sample.creds_key for sample in samples]
        samples = [sample.path for sample in samples]
        current_chunk = start_chunk
        updated_chunks = []
        if current_chunk is None:
            current_chunk = self._create_new_chunk(register)
            updated_chunks.append(current_chunk)
        enc = self.chunk_id_encoder
        link_creds = self.link_creds
        tiles = {}
        if register and update_commit_diff:
            commit_diff = self.commit_diff
        while len(samples) > 0:
            num_samples_added = current_chunk.extend_if_has_space(samples)  # type: ignore
            for i in range(num_samples_added):
                creds_key = creds[i]
                encoded_creds_key = link_creds.get_encoding(creds_key)
                self.creds_encoder.register_samples(encoded_creds_key, 1)
            if num_samples_added == 0:
                current_chunk = self._create_new_chunk(register)
                updated_chunks.append(current_chunk)
            else:
                if not updated_chunks:
                    updated_chunks.append(current_chunk)
                num = int(num_samples_added)
                if register:
                    enc.register_samples(num)
                    if update_commit_diff:
                        commit_diff.add_data(num)
                samples = samples[num:]
                creds = creds[num:]
        if register:
            return updated_chunks
        return updated_chunks, tiles
