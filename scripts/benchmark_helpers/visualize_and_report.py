import os
from visualizer import BenchmarkVisualizer
from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser(description='Visualize and report benchmark results')
parser.add_argument('--results_db',
    type=str,
    default='az://testactiveloop/indra-benchmarks/result-tables',
    help='Dataset containing benchmark results')

parser.add_argument('--current_results_dir',
    type=str,
    required=True,
    help='Result ID to compare and visualize')

args = parser.parse_args()

if __name__ == "__main__":
    path = Path(args.current_results_dir)
    session_id = ""
    with open(path / "ingestion.json", 'r') as f:
        data = json.load(f)
        session_id = data[0]["session_id"]
    if session_id == "":
        raise ValueError("No benchmark results found in the directory")

    visualizer = BenchmarkVisualizer(
        results_db=args.results_db,
        result_id=session_id,
    )

    reports = visualizer.generate_report()

    with open('benchmark_report.html', 'w') as f:
        f.write(reports['html'])
    with open('benchmark_summary.md', 'w') as f:
        f.write(reports['markdown'])

