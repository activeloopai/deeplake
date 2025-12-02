import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import deeplake

class BenchmarkVisualizer:
    def __init__(self, results_db: str, result_id: str):
        self.ingestion_db = deeplake.open_read_only(f'{results_db}/ingestion')
        self.queries_db = deeplake.open_read_only(f'{results_db}/queries')
        self.result_id = result_id
    
    def load_results(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Load current results and historical data"""
        current_reports = self._load_current_results()
        history_reports = self._load_historical_results()
        return current_reports, history_reports
    
    def _load_current_results(self) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
        """Load most recent benchmark results"""
        ingestion_reports = {}
        ingestion_view = self.ingestion_db.query(f"SELECT * WHERE session_id='{self.result_id}'")
        for data in ingestion_view:
            dataset = data['dataset']
            ingestion_reports[dataset] = {
                'ingestion_time': float(data['ingestion_time']),
                'ingestion_ram': float(data['ingestion_ram_usage']),
                'dataset_delete_time': float(data['dataset_delete_time']),
                'timestamp': float(data['timestamp']),
            }
        query_reports = {}
        query_view = self.queries_db.query(f"SELECT * WHERE session_id='{self.result_id}'")
        for data in query_view:
            dataset = data['dataset']
            if dataset not in query_reports:
                query_reports[dataset] = {}
            query_reports[dataset][data['query_type']] = {
                'load_time': float(data['load_time']),
                'query_time': float(data['query_time']),
                'query_ram': float(data['query_ram_usage']),
                'query_recall': float(data['query_recall']),
                'timestamp': float(data['timestamp']),
            }
        return ingestion_reports, query_reports
    
    def _load_historical_results(self) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, List[Dict]]]]:
        """Load historical results from the last 4 weeks"""
        history_cutoff = datetime.now() - timedelta(weeks=4)
        ingestion_view = self.ingestion_db.query(f"SELECT * WHERE session_id!='{self.result_id}' AND \"timestamp\">{history_cutoff.timestamp()} ORDER BY \"timestamp\"")
        jsons = []
        for data in ingestion_view:
            jsons.append(data)

        jsons.sort(key=lambda x: x["timestamp"])

        ingestion_reports = {}
        for data in jsons:
            dataset = data['dataset']
            if dataset not in ingestion_reports:
                ingestion_reports[dataset] = []

            historical_data = {
                'ingestion_time': float(data['ingestion_time']),
                'ingestion_ram': float(data['ingestion_ram_usage']),
                'dataset_delete_time': float(data['dataset_delete_time']),
                'timestamp': float(data['timestamp']),
            }

            ingestion_reports[dataset].append(historical_data)

        query_view = self.queries_db.query(f"SELECT * WHERE session_id!='{self.result_id}' AND \"timestamp\">{history_cutoff.timestamp()} ORDER BY \"timestamp\"")
        jsons = []
        for data in query_view:
            jsons.append(data)
        query_reports = {}
        for data in jsons:
            dataset = data['dataset']
            if dataset not in query_reports:
                query_reports[dataset] = {}

            if data['query_type'] not in query_reports[dataset]:
                query_reports[dataset][data['query_type']] = []
            historical_data = {
                'load_time': float(data['load_time']),
                'query_time': float(data['query_time']),
                'query_ram': float(data['query_ram_usage']),
                'query_recall': float(data['query_recall']),
                'timestamp': float(data['timestamp']),
            }

            query_reports[dataset][data['query_type']].append(historical_data)

        return ingestion_reports, query_reports


    def _generate_html_report(self, current_ingestion: Dict[str, Dict], current_query: Dict[str, Dict[str, Dict]], history_ingestion: Dict[str, List[Dict]], history_query: Dict[str, Dict[str, List[Dict]]]) -> str:
        """
        Generate the complete HTML report.
        The generated html has sidebar menu - Ingestion and Query.
        The ingestion section has the ingestion metrics for each dataset.
        The datasets are tabbed on top and each dataset tab has list of charts related to ingestion metrics.
        The query section has the query metrics for each dataset and each query type.
        The top level tab are the datasets, then for each dataset there are secondary tabs for each query type.
        Each query type content are charts of that query type and that dataset.
        """
        # Create the basic HTML structure with CSS
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Results</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.24.2/plotly.min.js"></script>
            <style>
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    min-height: 100vh;
                }
                .sidebar {
                    width: 250px;
                    background: #f4f4f4;
                    padding: 20px;
                    height: 100vh;
                    position: fixed;
                    overflow-y: auto;
                }
                .main-content {
                    margin-left: 250px;
                    padding: 20px;
                    width: calc(100% - 250px);
                    max-width: 100%;
                }
                .nav-link {
                    display: block;
                    padding: 10px;
                    color: #333;
                    text-decoration: none;
                    margin: 5px 0;
                    border-radius: 4px;
                    font-size: 16px;
                }
                .nav-link:hover, .nav-link.active {
                    background: #e0e0e0;
                }
                .tab-content {
                    display: none;
                    width: 100%;
                }
                .tab-content.active {
                    display: block;
                    width: 100%;
                }
                .dataset-tabs {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin: 20px 0;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 15px;
                }
                .dataset-tab {
                    padding: 12px 24px;
                    cursor: pointer;
                    border: none;
                    background: none;
                    border-radius: 4px;
                    font-size: 16px;
                }
                .dataset-tab:hover, .dataset-tab.active {
                    background: #e0e0e0;
                }
                .query-tabs {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin: 20px 0;
                }
                .dataset-content {
                    width: 100%;
                }
                .query-content {
                    width: 100%;
                }
                .charts-container {
                    display: flex;
                    flex-direction: column;
                    gap: 30px;
                    margin-top: 30px;
                    width: 100%;
                }
                .chart {
                    width: 100%;
                    height: 600px;
                    border: 1px solid #ddd;
                    padding: 20px;
                    border-radius: 8px;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h2 {
                    font-size: 24px;
                    margin: 20px 0;
                    color: #333;
                }

                /* Responsive Design */
                @media (max-width: 1024px) {
                    .sidebar {
                        width: 200px;
                    }
                    .main-content {
                        margin-left: 200px;
                        width: calc(100% - 200px);
                    }
                    .chart {
                        height: 500px;
                    }
                }

                @media (max-width: 768px) {
                    body {
                        flex-direction: column;
                    }
                    .sidebar {
                        width: 100%;
                        height: auto;
                        position: relative;
                    }
                    .main-content {
                        margin-left: 0;
                        width: 100%;
                    }
                    .dataset-tabs, .query-tabs {
                        overflow-x: auto;
                        padding-bottom: 10px;
                    }
                    .chart {
                        height: 400px;
                    }
                }
            </style>
        </head>
        <body>
        """

        # Add sidebar navigation
        html += """
        <div class="sidebar">
            <h2 style="margin-bottom: 20px;">Navigation</h2>
            <a href="#ingestion" class="nav-link" onclick="showSection('ingestion')">Ingestion Metrics</a>
            <a href="#queries" class="nav-link" onclick="showSection('queries')">Query Metrics</a>
        </div>
        """

        # Main content container
        html += '<div class="main-content">'

        # Ingestion Section
        html += '<div id="ingestion" class="tab-content active">'
        html += '<h2>Ingestion Metrics</h2>'

        # Dataset tabs for ingestion
        html += '<div class="dataset-tabs">'
        for i, dataset in enumerate(current_ingestion.keys()):
            active = 'active' if i == 0 else ''
            html += f'<button class="dataset-tab {active}" onclick="showDatasetTab(\'ingestion\', \'{dataset}\')">{dataset}</button>'
        html += '</div>'

        # Dataset content for ingestion
        for i, (dataset, metrics) in enumerate(current_ingestion.items()):
            display = 'block' if i == 0 else 'none'
            html += f'<div id="ingestion-{dataset}" class="dataset-content" style="display: {display};">'

            # Generate plots for ingestion metrics
            current_timestamp = datetime.fromtimestamp(metrics['timestamp'])
            history = history_ingestion.get(dataset, [])
            history_timestamps = [datetime.fromtimestamp(h['timestamp']) for h in history]

            metrics_list = ['ingestion_time', 'ingestion_ram', 'dataset_delete_time']
            html += '<div class="charts-container">'
            for metric in metrics_list:
                div_id = f'ingestion-{dataset}-{metric}'
                html += f'<div class="chart" id="{div_id}"></div>'

                # Generate Plotly figure
                fig = go.Figure()
                # Add current point
                fig.add_trace(go.Scatter(
                    x=[current_timestamp],
                    y=[metrics[metric]],
                    mode='markers',
                    name='Current',
                    marker=dict(size=12)
                ))
                # Add historical data
                if history:
                    fig.add_trace(go.Scatter(
                        x=history_timestamps,
                        y=[h[metric] for h in history],
                        mode='lines+markers',
                        name='History'
                    ))

                metric_name = metric.replace('_', ' ').title()
                fig.update_layout(
                    title=dict(
                        text=f"{dataset} {metric_name}",
                        font=dict(size=20)
                    ),
                    xaxis_title="Timestamp",
                    yaxis_title=metric_name,
                    height=500,
                    font=dict(size=12),
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    margin=dict(l=50, r=50, t=50, b=50)
                )

                # Add plot javascript
                html += f"<script>Plotly.newPlot('{div_id}', {fig.to_json()});</script>"

            html += '</div>'  # End charts-container
            html += '</div>'  # End dataset-content
        html += '</div>'  # End ingestion section

        # Queries Section
        html += '<div id="queries" class="tab-content">'
        html += '<h2>Query Metrics</h2>'

        # Dataset tabs for queries
        html += '<div class="dataset-tabs">'
        for i, dataset in enumerate(current_query.keys()):
            active = 'active' if i == 0 else ''
            html += f'<button class="dataset-tab {active}" onclick="showDatasetTab(\'queries\', \'{dataset}\')">{dataset}</button>'
        html += '</div>'

        # Dataset content for queries
        for i, (dataset, query_types) in enumerate(current_query.items()):
            display = 'block' if i == 0 else 'none'
            html += f'<div id="queries-{dataset}" class="dataset-content" style="display: {display};">'

            # Query type tabs
            html += '<div class="query-tabs">'
            for j, query_type in enumerate(query_types.keys()):
                active = 'active' if j == 0 else ''
                html += f'<button class="dataset-tab {active}" onclick="showQueryTab(\'{dataset}\', \'{query_type}\')">{query_type}</button>'
            html += '</div>'

            # Query type content
            for j, (query_type, metrics) in enumerate(query_types.items()):
                display = 'block' if j == 0 else 'none'
                html += f'<div id="query-{dataset}-{query_type}" class="query-content" style="display: {display};">'

                current_timestamp = datetime.fromtimestamp(metrics['timestamp'])
                history = history_query.get(dataset, {}).get(query_type, [])
                history_timestamps = [datetime.fromtimestamp(h['timestamp']) for h in history]

                metrics_list = ['load_time', 'query_time', 'query_ram', 'query_recall']
                html += '<div class="charts-container">'
                for metric in metrics_list:
                    div_id = f'query-{dataset}-{query_type}-{metric}'
                    html += f'<div class="chart" id="{div_id}"></div>'

                    # Generate Plotly figure
                    fig = go.Figure()
                    # Add current point
                    fig.add_trace(go.Scatter(
                        x=[current_timestamp],
                        y=[metrics[metric]],
                        mode='markers',
                        name='Current',
                        marker=dict(size=12)
                    ))
                    # Add historical data
                    if history:
                        fig.add_trace(go.Scatter(
                            x=history_timestamps,
                            y=[h[metric] for h in history],
                            mode='lines+markers',
                            name='History'
                        ))

                    metric_name = metric.replace('_', ' ').title()
                    fig.update_layout(
                        title=dict(
                            text=f"{dataset} {query_type} {metric_name}",
                            font=dict(size=20)
                        ),
                        xaxis_title="Timestamp",
                        yaxis_title=metric_name,
                        height=500,
                        font=dict(size=12),
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        ),
                        margin=dict(l=50, r=50, t=50, b=50)
                    )

                    # Add plot javascript
                    html += f"<script>Plotly.newPlot('{div_id}', {fig.to_json()});</script>"

                html += '</div>'  # End charts-container
                html += '</div>'  # End query-content
            html += '</div>'  # End dataset-content
        html += '</div>'  # End queries section

        html += '</div>'  # End main-content

        # Add JavaScript for navigation
        html += """
        <script>
        function showSection(section) {
            document.querySelectorAll('.tab-content').forEach(el => {
                el.classList.remove('active');
                el.style.display = 'none';
            });
            document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
            const sectionEl = document.getElementById(section);
            sectionEl.classList.add('active');
            sectionEl.style.display = 'block';
            document.querySelector(`a[href="#${section}"]`).classList.add('active');

            // Trigger resize for all visible charts
            sectionEl.querySelectorAll('.chart').forEach(chart => {
                Plotly.Plots.resize(chart);
            });
        }

        function showDatasetTab(section, dataset) {
            const contents = document.querySelectorAll(`#${section} .dataset-content`);
            contents.forEach(el => el.style.display = 'none');
            const datasetContent = document.getElementById(`${section}-${dataset}`);
            datasetContent.style.display = 'block';

            const tabs = document.querySelectorAll(`#${section} .dataset-tabs .dataset-tab`);
            tabs.forEach(tab => tab.classList.remove('active'));
            event.target.classList.add('active');

            // Trigger resize for all visible charts
            datasetContent.querySelectorAll('.chart').forEach(chart => {
                Plotly.Plots.resize(chart);
            });
        }

        function showQueryTab(dataset, queryType) {
            const contents = document.querySelectorAll(`#queries-${dataset} .query-content`);
            contents.forEach(el => el.style.display = 'none');
            const queryContent = document.getElementById(`query-${dataset}-${queryType}`);
            queryContent.style.display = 'block';

            const tabs = document.querySelectorAll(`#queries-${dataset} .query-tabs .dataset-tab`);
            tabs.forEach(tab => tab.classList.remove('active'));
            event.target.classList.add('active');

            // Trigger resize for all visible charts
            queryContent.querySelectorAll('.chart').forEach(chart => {
                Plotly.Plots.resize(chart);
            });
        }

        // Make charts responsive
        window.addEventListener('resize', function() {
            document.querySelectorAll('.chart').forEach(function(chart) {
                if (chart.offsetParent !== null) { // Only resize visible charts
                    Plotly.Plots.resize(chart);
                }
            });
        });

        // Initial resize of visible charts
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.chart').forEach(function(chart) {
                if (chart.offsetParent !== null) {
                    Plotly.Plots.resize(chart);
                }
            });
        });
        </script>
        </body>
        </html>
        """

        return html


    def _generate_dataset_report(self,
                               dataset: str,
                               current_ingestion: Dict,
                               history_ingestion: List[Dict],
                               current_query: Dict = None,
                               history_query: Dict[str, List[Dict]] = None) -> Dict[str, go.Figure]:
        """Generate visualizations for a specific dataset"""
        report = {}

        # Convert timestamps to datetime objects for ingestion data
        current_timestamp = datetime.fromtimestamp(current_ingestion['timestamp'])
        history_timestamps = [datetime.fromtimestamp(df['timestamp']) for df in history_ingestion]

        # Ingestion metrics (keep existing code)
        # ... (keep existing ingestion metric generation code) ...

        # Add query metrics if available
        if current_query and history_query:
            for query_type, current_metrics in current_query.items():
                history_metrics = history_query.get(query_type, [])
                
                # Convert timestamps for query data
                query_current_timestamp = datetime.fromtimestamp(current_metrics['timestamp'])
                query_history_timestamps = [datetime.fromtimestamp(m['timestamp']) for m in history_metrics]

                # Generate plots for each query metric
                for metric in ['load_time', 'query_time', 'query_ram', 'query_recall']:
                    fig = go.Figure()
                    
                    # Add current point
                    fig.add_trace(go.Scatter(
                        x=[query_current_timestamp],
                        y=[current_metrics[metric]],
                        mode='markers',
                        name='Current'
                    ))
                    
                    # Add historical data if available
                    if history_metrics:
                        fig.add_trace(go.Scatter(
                            x=query_history_timestamps,
                            y=[m[metric] for m in history_metrics],
                            mode='lines+markers',
                            name='History'
                        ))
                    
                    # Update layout
                    metric_name = metric.replace('_', ' ').title()
                    fig.update_layout(
                        title=f"{dataset} {query_type} {metric_name}",
                        xaxis_title="Timestamp",
                        yaxis_title=metric_name
                    )
                    
                    # Store in report
                    report[f"{query_type}_{metric}"] = fig

        return report


    def generate_report(self) -> Dict[str, str]:
        """Generate HTML report with per-dataset visualizations and regression analysis"""
        (current_ingestion_reports, current_query_reports), (history_ingestion_reports, history_query_reports) = self.load_results()

        # Generate HTML
        report_html = self._generate_html_report(current_ingestion_reports, current_query_reports, history_ingestion_reports, history_query_reports)

        # Generate markdown summary
        markdown_summary = self._generate_markdown_summary(current_ingestion_reports, current_query_reports)

        return {
            'html': report_html,
            'markdown': markdown_summary,
        }

    def _generate_markdown_summary(self, ingestion_reports: Dict[str, Dict], query_reports: Dict[str, Dict[str, Dict]]) -> str:
        """Generate markdown summary with per-dataset sections"""
        markdown = "# Benchmark Report Summary\n\n"

        for dataset, metrics in ingestion_reports.items():
            markdown += f"## {dataset}\n\n"

            markdown += "### Ingestion Metrics\n"
            markdown += f"- Ingestion Time: {metrics['ingestion_time']:.2f} s\n"
            markdown += f"- Ingestion RAM Usage: {metrics['ingestion_ram']:.2f} MB\n"
            markdown += f"- Dataset Delete Time: {metrics['dataset_delete_time']:.2f} s\n\n"

        return markdown
