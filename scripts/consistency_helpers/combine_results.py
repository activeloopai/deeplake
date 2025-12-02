#!/usr/bin/env python3
"""
Combine consistency test results into a single JSON file and generate a summary report.
"""

import json
import glob
import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    # Get run ID from command line argument
    run_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    
    # Find all result files
    result_files = glob.glob("all-results/*.json")
    
    combined_results = {
        "timestamp": int(datetime.now().timestamp()),
        "run_id": run_id,
        "configurations": []
    }
    
    for result_file in result_files:
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            combined_results["configurations"].append(data)
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    # Write combined results
    with open("combined_consistency_results.json", "w") as f:
        json.dump(combined_results, f, indent=2)
    
    # Generate summary report
    summary = "# Combined Consistency Test Results\n\n"
    summary += f"**Run ID:** {run_id}\n"
    summary += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
    
    for config_data in combined_results["configurations"]:
        # Add cloud provider info
        cloud_provider = config_data.get("cloud_provider", "unknown")
        summary += f"**Cloud Provider:** {cloud_provider}\n\n"
        
        for metric in config_data.get("metrics", []):
            workers = metric.get("num_workers", 0)
            ops = metric.get("operations_per_worker", 0)
            
            # Determine config name based on parameters
            if workers == 3 and ops == 10:
                config_name = "Light Load"
            elif workers == 10 and ops == 50:
                config_name = "Medium Load"
            elif workers == 24 and ops == 600:
                config_name = "Heavy Load"
            else:
                config_name = f"{workers}w-{ops}ops"
            
            summary += f"## {config_name} Configuration\n"
            summary += f"- **Workers:** {workers}\n"
            summary += f"- **Operations per worker:** {ops}\n"
            summary += f"- **Initial rows:** {metric.get('initial_dataset_size', 'N/A')}\n"
            summary += f"- **Total runtime:** {metric.get('total_runtime', 0):.2f} seconds\n"
            summary += f"- **Success rate:** {metric.get('success_rate', 0):.1f}%\n"
            
            consistent = metric.get("dataset_consistent", False)
            consistent_icon = "✅" if consistent else "❌"
            summary += f"- **Dataset consistent:** {consistent_icon} {consistent}\n"
            
            crashed = metric.get("crashed_workers", 0)
            crashed_icon = "⚠️" if crashed > 0 else "✅"
            summary += f"- **Crashed workers:** {crashed_icon} {crashed}\n"
            summary += f"- **Memory usage:** {metric.get('memory_usage', 0):.2f} GB\n"
            
            errors = metric.get("errors", [])
            if errors:
                summary += f"- **Errors:** 🚨 {len(errors)}\n"
                # Check for retained dataset paths
                retained_datasets = [e for e in errors if "Dataset retained at:" in e]
                if retained_datasets:
                    summary += f"- **Retained datasets:** 🔍 {len(retained_datasets)} (for debugging)\n"
            else:
                summary += "- **Errors:** ✅ None\n"
            summary += "\n"
    
    with open("combined_consistency_summary.md", "w") as f:
        f.write(summary)
    
    print("Combined results and summary generated successfully")

if __name__ == "__main__":
    main() 