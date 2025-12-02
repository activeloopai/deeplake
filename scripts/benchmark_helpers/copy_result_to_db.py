import argparse
import deeplake
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='Copy result to database')

parser.add_argument('--results_dir', type=str, help='results directory')
parser.add_argument('--results_db',
    type=str,
    default='az://testactiveloop/indra-benchmarks/result-tables',
    help='Path to datasets containing the results')

args = parser.parse_args()

if __name__ == "__main__":
    path = Path(args.results_dir)
    if not deeplake.exists(f'{args.results_db}/ingestion'):
        ds = deeplake.create(f'{args.results_db}/ingestion')
        ds.add_column("dataset", "text")
        ds.add_column("indra_branch", "text")
        ds.add_column("indra_commit", "text")
        ds.add_column("timestamp", "float32")
        ds.add_column("session_id", "text")
        ds.add_column("ingestion_time", "float32")
        ds.add_column("ingestion_ram_usage", "float32")
        ds.add_column("dataset_delete_time", "float32")
        ds.commit()
    ds = deeplake.open(f'{args.results_db}/ingestion')
    with open(path / "ingestion.json", 'r') as f:
        jj = json.load(f)
        for data in jj:
            ds.append([{
                "dataset": data["dataset"],
                "indra_branch": data["indra_branch"],
                "indra_commit": data["indra_commit"],
                "timestamp": data["timestamp"],
                "session_id": data["session_id"],
                "ingestion_time": data["ingestion_time"],
                "ingestion_ram_usage": data["ingestion_ram_usage"],
                "dataset_delete_time": data["delete_time"]
            }])
        ds.commit()


    if not deeplake.exists(f'{args.results_db}/queries'):
        ds = deeplake.create(f'{args.results_db}/queries')
        ds.add_column("dataset", "text")
        ds.add_column("indra_branch", "text")
        ds.add_column("indra_commit", "text")
        ds.add_column("timestamp", "float32")
        ds.add_column("session_id", "text")
        ds.add_column("load_time", "float32")
        ds.add_column("query_type", "text")
        ds.add_column("query_string", "text")
        ds.add_column("query_time", "float32")
        ds.add_column("query_ram_usage", "float32")
        ds.add_column("query_recall", "float32")
        ds.commit()
    ds = deeplake.open(f'{args.results_db}/queries')
    with open(path / "queries.json", 'r') as f:
        jj = json.load(f)
        for data in jj:
            ds.append([{
                "dataset": data["dataset"],
                "indra_branch": data["indra_branch"],
                "indra_commit": data["indra_commit"],
                "timestamp": data["timestamp"],
                "session_id": data["session_id"],
                "load_time": data["load_time"],
                "query_type": data["query_type"],
                "query_string": data["query_string"],
                "query_time": data["time"],
                "query_ram_usage": data["ram_usage"],
                "query_recall": data["recall"]
            }])
        ds.commit()
