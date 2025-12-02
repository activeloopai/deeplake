#!/usr/bin/env python3
"""
Helper script for running Deep Lake consistency tests locally.

This script provides an easy way to run consistency tests with different
configurations and automatically generate reports.
"""

import argparse
import sys
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import time


def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    return result


def generate_summary_report(results_dir):
    """Generate a summary report from consistency test results."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return
    
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    # Load the latest result file
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        print("\n" + "="*60)
        print("CONSISTENCY TEST SUMMARY")
        print("="*60)
        print(f"Session ID: {data.get('session_id', 'N/A')}")
        print(f"Timestamp: {datetime.fromtimestamp(data.get('timestamp', 0))}")
        
        for metric in data.get('metrics', []):
            print(f"\n📊 Test: {metric.get('test_name', 'Unknown')}")
            print("-" * 40)
            print(f"Workers: {metric.get('num_workers', 0)}")
            print(f"Operations per worker: {metric.get('operations_per_worker', 0)}")
            print(f"Total runtime: {metric.get('total_runtime', 0):.2f} seconds")
            print(f"Success rate: {metric.get('success_rate', 0):.1f}%")
            
            consistent = metric.get('dataset_consistent', False)
            status_emoji = "✅" if consistent else "❌"
            print(f"Dataset consistent: {status_emoji} {consistent}")
            
            crashed = metric.get('crashed_workers', 0)
            if crashed > 0:
                print(f"⚠️  Crashed workers: {crashed}")
            else:
                print(f"Crashed workers: ✅ {crashed}")
            
            print(f"Memory usage: {metric.get('memory_usage', 0):.2f} GB")
            
            errors = metric.get('errors', [])
            if errors:
                print(f"🚨 Errors ({len(errors)}):")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"  - {error}")
                if len(errors) > 3:
                    print(f"  ... and {len(errors) - 3} more")
            else:
                print("Errors: ✅ None")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error reading results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Deep Lake consistency tests with various configurations"
    )
    
    # Test configuration
    parser.add_argument(
        "--workers", "-w", 
        type=int, 
        default=10,
        help="Number of concurrent workers (default: 10)"
    )
    parser.add_argument(
        "--operations", "-o",
        type=int,
        default=300,
        help="Operations per worker (default: 300)"
    )
    parser.add_argument(
        "--initial-rows", "-i",
        type=int,
        default=5000,
        help="Initial number of rows in dataset (default: 5000)"
    )
# Note: Heavy load tests removed - use different parameters instead
    parser.add_argument(
        "--results-dir", "-r",
        default="consistency_results",
        help="Directory for test results (default: consistency_results)"
    )
    
    # Test selection
    parser.add_argument(
        "--test", "-t",
        choices=["concurrent", "heavy", "all"],
        default="all",
        help="Which test to run (default: all)"
    )
    
    # Predefined configurations
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: 3 workers, 50 operations"
    )
    parser.add_argument(
        "--stress",
        action="store_true", 
        help="Stress test: 30 workers, 1000 operations"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating summary report"
    )
    
    args = parser.parse_args()
    
    # Handle predefined configurations
    if args.quick:
        args.workers = 3
        args.operations = 50
        args.initial_rows = 1000
        print("🚀 Running quick consistency test...")
    elif args.stress:
        args.workers = 50
        args.operations = 1000
        args.initial_rows = 10000
        print("💪 Running stress consistency test...")
    
    # Change to python directory
    script_dir = Path(__file__).parent
    python_dir = script_dir.parent / "python"
    
    if not python_dir.exists():
        print(f"Error: Python directory not found at {python_dir}")
        sys.exit(1)
    
    os.chdir(python_dir)
    print(f"Working directory: {python_dir}")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Select test
    if args.test == "concurrent":
        cmd.append("tests/consistency/consistency_test.py::test_concurrent_access")
    elif args.test == "heavy":
        cmd.append("tests/consistency/consistency_test.py::test_heavy_load_concurrent_access")
    else:
        cmd.extend(["tests/consistency/", "-m", "consistency"])
    
    # Add options
    if args.verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-v")
    
    cmd.extend([
        f"--consistency_num_workers={args.workers}",
        f"--consistency_operations_per_worker={args.operations}",
        f"--consistency_initial_rows={args.initial_rows}",
        f"--consistency_results_dir={args.results_dir}"
    ])
    
    # Run the test
    print(f"\n🧪 Running consistency tests...")
    print(f"Workers: {args.workers}")
    print(f"Operations per worker: {args.operations}")
    print(f"Initial rows: {args.initial_rows}")
    print(f"Results directory: {args.results_dir}")
    
    start_time = time.time()
    result = run_command(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"\n⏱️  Total test time: {end_time - start_time:.2f} seconds")
    
    # Generate report
    if not args.no_report:
        print("\n📋 Generating summary report...")
        generate_summary_report(args.results_dir)
    
    # Exit with test result code
    if result.returncode == 0:
        print("\n✅ All consistency tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {result.returncode}")
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main() 