import deeplake
import numpy as np

from deeplake.core.distance_type import DistanceType
from deeplake.tests.common import requires_libdeeplake
from deeplake.tests.dataset_fixtures import local_auth_ds_generator
from deeplake.util.exceptions import ReadOnlyModeError, EmbeddingTensorPopError
import pytest
import warnings

statements = [
    "The apple fell from the tree and rolled into the river.",
    "In the jungle, the giraffe munched on the leaves of a tall tree.",
    "A rainbow appeared in the sky after the rain stopped.",
    "The computer screen flickered as the storm intensified outside.",
    "She found a book about coding with Python on the dusty shelf.",
    "As the sun set, the mountain peaks were bathed in orange light.",
    "The cat jumped onto the window sill to watch the birds outside.",
    "He poured himself a cup of coffee and stared out at the ocean.",
    "The children played under the table, laughing and giggling.",
    "With a splash, the dog jumped into the river to fetch the stick.",
]

new_statements = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a widely used high-level programming language.",
    "The sun shines bright over the tall mountains.",
    "A cup of tea is a good companion while coding.",
    "The river flowed swiftly under the wooden bridge.",
    "In autumn, the leaves fall gently to the ground.",
    "The stars twinkled brightly in the night sky.",
    "She prefers coffee over tea during the early hours.",
    "The code compiled successfully without any errors.",
    "Birds fly south for the winter every year.",
]

statements1 = [
    "The text is about the apple falling from the tree.",
    "Finally searched the jungle for the giraffe.",
]


@requires_libdeeplake
def test_inv_index_(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        for statement in statements:
            ds.text.append(statement)

        ds.summary()

        # create inverted index.
        ds.text.create_vdb_index("inv_1")
        ts = ds.text.get_vdb_indexes()
        assert len(ts) == 1
        assert ts[0]["id"] == "inv_1"

        # drop the inverted index.
        ds.text.delete_vdb_index("inv_1")
        ts = ds.text.get_vdb_indexes()
        assert len(ts) == 0


@requires_libdeeplake
def test_inv_index_query(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        for statement in statements:
            ds.text.append(statement)

        # create inverted index.
        ds.text.create_vdb_index("inv_1")

        # query the inverted index this should fail as equalities are not supported.
        res = ds.query(f"select * where text == 'apple'")
        assert len(res) == 0

        # query the inverted index.
        res = ds.query(f"select * where CONTAINS(text, 'flickered')")
        assert len(res) == 1
        assert res.index[0].values[0].value == 3

        # query the inverted index.
        res = ds.query(f"select * where CONTAINS(text, 'mountain')")
        assert len(res) == 1
        assert res.index[0].values[0].value == 5

        # query the inverted index.
        res = ds.query(f"select * where CONTAINS(text, 'mountain')")
        assert len(res) == 1
        assert res.index[0].values[0].value == 5

        # query the inverted index.
        res = ds.query(f"select * where CONTAINS(text, 'jumped')")
        assert len(res) == 2
        assert res.index[1].values[0].value == 9

        ds.text.unload_vdb_index_cache()


@requires_libdeeplake
def test_inv_index_query_with_hnsw(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for statement in statements:
            random_embedding = np.random.random_sample(384).astype(np.float32)
            ds.append({"text": statement, "embedding": random_embedding})

        print(ds.text[2].numpy())
        # create inverted index.
        ds.text.create_vdb_index("inv_1")
        ds.embedding.create_vdb_index("hnsw_1")

        # query the inverted index along with hnsw index.
        v2 = ds.embedding[0].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)
        res = ds.query(
            f"select * where CONTAINS(text, 'apple') order by l2_norm(embedding - ARRAY[{s2}]) limit 1"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 0

        # query the inverted index along with hnsw index.
        v2 = ds.embedding[5].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)
        res = ds.query(
            f"select * where CONTAINS(text, 'mountain') order by l2_norm(embedding - ARRAY[{s2}]) limit 1"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 5

        # query the inverted index along with hnsw index.
        v2 = ds.embedding[9].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)
        res = ds.query(
            f"select * where CONTAINS(text, 'jumped') order by l2_norm(embedding - ARRAY[{s2}]) limit 1"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 9

        ds.text.unload_vdb_index_cache()
        ds.embedding.unload_vdb_index_cache()


@requires_libdeeplake
def test_inv_index_multiple_where_or_and(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        for statement in statements + new_statements:
            ds.text.append(statement)

        # create inverted index.
        ds.text.create_vdb_index("inv_1")

        # query with multiple WHERE clauses using OR and AND
        res = ds.query(
            f"select * where CONTAINS(text, 'mountains.') and CONTAINS(text, 'bright')"
        )
        print(res)
        assert len(res) == 1
        assert (
            res.index[0].values[0].value == 12
        )  # "The sun shines bright over the tall mountains."

        ds.text.unload_vdb_index_cache()


@requires_libdeeplake
def test_inv_index_multiple_keywords(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        for statement in statements + new_statements:
            ds.text.append(statement)

        # create inverted index.
        ds.text.create_vdb_index("inv_1")

        # query with multiple keywords in WHERE clause
        res = ds.query(
            f"select * where CONTAINS(text, 'sun') and CONTAINS(text, 'bright')"
        )
        assert len(res) == 1
        assert (
            res.index[0].values[0].value == 12
        )  # "The sun shines bright over the tall mountains."

        res = ds.query(
            f"select * where CONTAINS(text, 'quick') and CONTAINS(text, 'fox')"
        )
        assert len(res) == 1
        assert (
            res.index[0].values[0].value == 10
        )  # "The quick brown fox jumps over the lazy dog."

        ds.text.unload_vdb_index_cache()


@requires_libdeeplake
def test_inv_index_case_insensitivity(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        for statement in statements + new_statements:
            ds.text.append(statement)

        # create inverted index.
        ds.text.create_vdb_index("inv_1")

        # query with case insensitivity
        res = ds.query(
            f"select * where CONTAINS(text, 'SUN')"
        )  # Case insensitive match
        assert len(res) == 2
        assert (
            res.index[0].values[0].value == 5
        )  # "The sun shines bright over the tall mountains."

        res = ds.query(
            f"select * where CONTAINS(text, 'PYTHON')"
        )  # Case insensitive match
        assert len(res) == 2
        assert (
            res.index[0].values[0].value == 4
        )  # "Python is a widely used high-level programming language."

        ds.text.unload_vdb_index_cache()


@requires_libdeeplake
def test_multiple_where_clauses_and_filters(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        ds.create_tensor("text1", htype="text")
        ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        ds.create_tensor("year", htype="text")  # Changed to text

        for i, statement in enumerate(statements):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            ds.append(
                {
                    "text": statement,
                    "text1": " ".join(
                        statement.split()[-3:]
                    ),  # last 3 words as a separate column
                    "embedding": random_embedding,
                    "year": str(
                        2015 + (i % 7)
                    ),  # cycles between 2015 and 2021 as strings
                }
            )

        # Create inverted index on text and year
        ds.text.create_vdb_index("inv_1")
        ds.year.create_vdb_index("inv_year")

        # Test 1: Multiple WHERE clauses with OR and AND filters
        res = ds.query(
            f"select * where CONTAINS(text, 'river.') and CONTAINS(year, '2015')"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 0  # 'apple' is in the first statement

        res = ds.query(
            f"select * where CONTAINS(text, 'rainbow') or CONTAINS(text1, 'rain') and CONTAINS(year, '2017')"
        )
        assert len(res) == 1
        assert (
            res.index[0].values[0].value == 2
        )  # 'rainbow' matches in the third statement

        # Test 2: Multiple keywords in WHERE clause
        res = ds.query(
            f"select * where CONTAINS( text, 'apple') and CONTAINS( text, 'tree')"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 0

        res = ds.query(
            f"select * where CONTAINS( text, 'apple') or CONTAINS(text, 'river.')"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 0

        res = ds.query(
            f"select * where CONTAINS(text, 'coding') and CONTAINS(year, '2019')"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 4

        # Test 3: Case insensitivity
        res = ds.query(f"select * where CONTAINS(text, 'Apple')")
        assert len(res) == 1
        assert res.index[0].values[0].value == 0

        res = ds.query(f"select * where CONTAINS(text, 'Tree')")
        assert len(res) == 1

        ds.text.unload_vdb_index_cache()
        ds.year.unload_vdb_index_cache()


@requires_libdeeplake
def test_hnsw_order_by_clause(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        ds.create_tensor("embedding", htype="embedding", dtype=np.float32)

        for statement in statements:
            random_embedding = np.random.random_sample(384).astype(np.float32)
            ds.append({"text": statement, "embedding": random_embedding})

        # Create inverted index and HNSW index
        ds.text.create_vdb_index("inv_1")
        ds.embedding.create_vdb_index("hnsw_1")

        # Test 4: ORDER BY clause with HNSW
        v2 = ds.embedding[5].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)
        res = ds.query(
            f"select * where CONTAINS(text, 'sun') order by l2_norm(embedding - ARRAY[{s2}]) limit 1"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 5

        # failure case.
        v2 = ds.embedding[5].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)
        res = ds.query(
            f"select * where text == 'sun' order by l2_norm(embedding - ARRAY[{s2}]) limit 1"
        )
        assert len(res) == 0

        ds.text.unload_vdb_index_cache()
        ds.embedding.unload_vdb_index_cache()


@requires_libdeeplake
def test_where_condition_on_column_without_inverted_index(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        ds.create_tensor("text1", htype="text")
        ds.create_tensor("embedding", htype="embedding", dtype=np.float32)

        for i, statement in enumerate(statements):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            ds.append(
                {
                    "text": statement,
                    "text1": " ".join(
                        statement.split()[-3:]
                    ),  # last 3 words as a separate column
                    "embedding": random_embedding,
                }
            )

        # Create inverted index on text only
        ds.text.create_vdb_index("inv_1")
        ds.embedding.create_vdb_index("hnsw_1")

        # res = ds.query(f"select * where CONTAINS(text, 'sun') and CONTAINS(text, 'bright')")

        # Test 5: WHERE condition on a column without an inverted index
        v2 = ds.embedding[0].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)
        res = ds.query(
            f"select * where CONTAINS(text, 'fell') or CONTAINS(text, 'river.') or CONTAINS(text, 'rolled') order by l2_norm(embedding - ARRAY[{s2}]) limit 1"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 0

        ds.text.unload_vdb_index_cache()
        ds.embedding.unload_vdb_index_cache()


@requires_libdeeplake
def test_multiple_where_clauses_and_filters_with_year_text(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        ds.create_tensor("text1", htype="text")
        ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        ds.create_tensor("year", htype="text")  # Changed to text type

        years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]

        for i, statement in enumerate(statements):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            ds.append(
                {
                    "text": statement,
                    "text1": " ".join(
                        statement.split()[-5:]
                    ),  # last 3 words as a separate column
                    "embedding": random_embedding,
                    "year": years[i % len(years)],  # cycles between 2015 and 2021
                }
            )

        # Create inverted index on text and year
        ds.text.create_vdb_index("inv_1")
        ds.year.create_vdb_index("inv_year")

        # Test 1: Multiple WHERE clauses with OR and AND filters
        res = ds.query(
            f"select * where CONTAINS(text, 'apple') or CONTAINS(text1, 'river.') and CONTAINS(year, '2016')"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 0  # 'apple' is in the first statement

        res = ds.query(
            f"select * where CONTAINS(text, 'rainbow') or CONTAINS(text1, 'dusty')"
        )
        assert len(res) == 2
        assert res.index[0].values[0].value == 2
        assert res.index[1].values[0].value == 4

        # Test 2: Multiple keywords in WHERE clause
        res = ds.query(
            f"select * where CONTAINS(text, 'apple') or CONTAINS(text1, 'river')"
        )
        assert len(res) == 2
        assert res.index[0].values[0].value == 0
        assert res.index[1].values[0].value == 9

        res = ds.query(
            f"select * where CONTAINS(text, 'coding') or CONTAINS(year, '2020') or CONTAINS(year, '2021') or CONTAINS(year, '2019')"
        )
        assert len(res) == 3

        res = ds.query(
            f"select * where CONTAINS(text, 'coding') or CONTAINS(year, '2020') or CONTAINS(year, '2021') or CONTAINS(year, '2018')"
        )
        assert len(res) == 4

        ds.text.unload_vdb_index_cache()
        ds.year.unload_vdb_index_cache()


@requires_libdeeplake
def test_inverted_index_on_year_column_with_text(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        ds.create_tensor("year", htype="text")
        ds.create_tensor("embedding", htype="embedding", dtype=np.float32)

        years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]

        for i, statement in enumerate(statements):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            ds.append(
                {
                    "text": statement,
                    "year": years[
                        i % len(years)
                    ],  # cycles between 2015 and 2021 as strings
                    "embedding": random_embedding,
                }
            )

        # Create inverted index on year only
        ds.year.create_vdb_index("inv_year")
        ds.embedding.create_vdb_index("hnsw_1")

        # Test: Multiple OR conditions on year and embedding column
        v2 = ds.embedding[5].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)
        res = ds.query(
            f"select * where CONTAINS(year, '2019') or CONTAINS(year, '2020') or CONTAINS(year, '2021') order by l2_norm(embedding - ARRAY[{s2}]) limit 1"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 5

        ds.year.unload_vdb_index_cache()
        ds.embedding.unload_vdb_index_cache()


@requires_libdeeplake
def test_inverted_index_regeneration(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        for statement in statements:
            ds.text.append(statement)

        # create inverted index.
        ds.text.create_vdb_index("inv_1")

        # query the inverted index.
        res = ds.query(f"select * where CONTAINS(text, 'flickered')")
        assert len(res) == 1
        assert res.index[0].values[0].value == 3

        for statement in statements1:
            ds.text.append(statement)

        # query the inverted index.
        res = ds.query(f"select * where CONTAINS(text, 'flickered')")
        assert len(res) == 1
        assert res.index[0].values[0].value == 3

        res = ds.query(f"select * where CONTAINS(text, 'searched')")
        assert len(res) == 1
        assert res.index[0].values[0].value == 11

        ds.text.unload_vdb_index_cache()


@requires_libdeeplake
def test_inverted_index_multiple_tensor_maintenance(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text")
        ds.create_tensor("year", htype="text")
        ds.create_tensor("embedding", htype="embedding", dtype=np.float32)

        years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
        years1 = ["2022", "2023"]

        for i, statement in enumerate(statements):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            ds.append(
                {
                    "text": statement,
                    "year": years[
                        i % len(years)
                    ],  # cycles between 2015 and 2021 as strings
                    "embedding": random_embedding,
                }
            )

        # Create inverted index on year only
        ds.year.create_vdb_index("inv_year")
        ds.embedding.create_vdb_index("hnsw_1")

        # Test: Multiple OR conditions on year and embedding column
        v2 = ds.embedding[5].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)
        res = ds.query(
            f"select * where CONTAINS(year, '2019') or CONTAINS(year, '2020') or CONTAINS(year, '2021')  order by cosine_similarity(embedding, ARRAY[{s2}]) DESC limit 1"
        )
        assert len(res) == 1
        assert res.index[0].values[0].value == 5

        for i, statement in enumerate(statements1):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            ds.append(
                {
                    "text": statement,
                    "year": years1[i % len(years1)],
                    "embedding": random_embedding,
                }
            )
        v2 = ds.embedding[11].numpy(fetch_chunks=True)
        s2 = ",".join(str(c) for c in v2)

        res = ds.query(
            f"select * where CONTAINS(year, '2023') order by cosine_similarity(embedding, ARRAY[{s2}]) DESC limit 1"
        )

        assert len(res) == 1
        assert res.index[0].values[0].value == 11

        ds.year.unload_vdb_index_cache()
        ds.embedding.unload_vdb_index_cache()
