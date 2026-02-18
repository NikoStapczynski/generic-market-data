"""
graph_generator.py - Enhanced Graph Generation Module for Compgrapher

This module provides configurable graph generation with statistics calculation,
multiple output formats, and comprehensive reporting capabilities.
"""
import io
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    """Configuration for graph appearance and behavior"""
    
    # Colors
    client_color: str = '#AAA'
    default_color: str = '#FFF'
    edge_color: str = 'black'
    grid_color: str = '#AAA'
    
    # Figure settings
    figure_width: float = 10.0
    figure_height: float = 6.0
    dpi: int = 100
    
    # Labels
    xlabel: str = 'Location'
    ylabel: str = 'Pay Range (Hourly)'
    
    # Grid and styling
    show_grid: bool = True
    show_labels: bool = False
    bar_width: float = 0.8
    
    # Output settings
    output_dir: str = 'output'
    supported_formats: List[str] = field(default_factory=lambda: [
        'html', 'pdf', 'png', 'svg', 'jpg', 'jpeg', 'webp', 'eps'
    ])


@dataclass 
class PositionStats:
    """Statistics for a single position"""
    position_name: str
    employer_count: int
    min_salary: float
    max_salary: float
    median_salary: float
    mean_salary: float
    salary_range: float
    
    def to_dict(self) -> Dict:
        return {
            'position': self.position_name,
            'employers': self.employer_count,
            'min': self.min_salary,
            'max': self.max_salary,
            'median': self.median_salary,
            'mean': self.mean_salary,
            'range': self.salary_range
        }


class GraphGenerator:
    """
    Enhanced graph generator with statistics and multiple output formats.
    
    Attributes:
        config: GraphConfig instance for customization
        stats: Dictionary of position statistics
    """
    
    # Column name constants
    LOCATION = 'location'
    TITLE = 'POSITION TITLE'
    SAL_MIN = 'salary_min'
    SAL_MAX = 'salary_max'
    COLOR = 'color'
    
    def __init__(self, config: Optional[GraphConfig] = None):
        """
        Initialize the graph generator.
        
        Args:
            config: Optional GraphConfig for customization
        """
        self.config = config or GraphConfig()
        self.stats: Dict[str, PositionStats] = {}
        logger.info("GraphGenerator initialized with config")
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, PositionStats]:
        """
        Calculate statistics for all positions in the data.
        
        Args:
            df: DataFrame with salary data
            
        Returns:
            Dictionary mapping position names to PositionStats
        """
        logger.info("Calculating statistics for all positions")
        stats = {}
        
        for position_name, group in df.groupby(self.TITLE):
            all_salaries = pd.concat([group[self.SAL_MIN], group[self.SAL_MAX]])
            
            stats[position_name] = PositionStats(
                position_name=position_name,
                employer_count=len(group),
                min_salary=group[self.SAL_MIN].min(),
                max_salary=group[self.SAL_MAX].max(),
                median_salary=all_salaries.median(),
                mean_salary=all_salaries.mean(),
                salary_range=group[self.SAL_MAX].max() - group[self.SAL_MIN].min()
            )
        
        self.stats = stats
        logger.info(f"Calculated statistics for {len(stats)} positions")
        return stats
    
    def generate_graphs(
        self,
        df: pd.DataFrame,
        output_formats: List[str],
        client_name: str,
        input_file: str,
        show_progress: bool = False
    ) -> List[str]:
        """
        Generate graphs in multiple formats.
        
        Args:
            df: DataFrame with salary data
            output_formats: List of output format strings
            client_name: Name of the client to highlight
            input_file: Path to input file (for HTML report)
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated file paths
        """
        logger.info(f"Generating graphs in formats: {output_formats}")
        generated_files = []
        
        # Calculate statistics
        self.calculate_statistics(df)
        
        # Separate image formats from HTML
        image_formats = [fmt for fmt in output_formats if fmt != 'html']
        
        # Generate image graphs
        if image_formats:
            files = self._generate_image_graphs(df, image_formats, show_progress)
            generated_files.extend(files)
        
        # Generate HTML report
        if 'html' in output_formats:
            html_file = self._generate_html_report(df, client_name, input_file)
            generated_files.append(html_file)
        
        logger.info(f"Generated {len(generated_files)} files")
        return generated_files
    
    def _generate_image_graphs(
        self,
        df: pd.DataFrame,
        formats: List[str],
        show_progress: bool = False
    ) -> List[str]:
        """Generate image format graphs"""
        generated_files = []
        groups = list(df.groupby(self.TITLE))
        
        for i, (name, group) in enumerate(groups):
            if show_progress:
                logger.info(f"Processing {i+1}/{len(groups)}: {name}")
            
            safe_name = name.replace('/', '_')
            group = group.sort_values(by=self.SAL_MAX, ascending=True)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
            
            # Calculate bar heights
            heights = group[self.SAL_MAX] - group[self.SAL_MIN]
            linewidths = [3 if h == 0 else 1 for h in heights]
            
            # Draw bars
            bars = ax.bar(
                group[self.LOCATION],
                heights,
                bottom=group[self.SAL_MIN],
                color=group[self.COLOR],
                edgecolor=self.config.edge_color,
                linewidth=linewidths,
                zorder=3,
                width=self.config.bar_width
            )
            
            # Add labels if configured
            if self.config.show_labels:
                for bar, (_, row) in zip(bars, group.iterrows()):
                    height = bar.get_height()
                    ax.annotate(
                        f'${row[self.SAL_MAX]:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=8
                    )
            
            # Configure axes
            ax.set_ylabel(self.config.ylabel)
            ax.set_xlabel(self.config.xlabel)
            ax.set_title(name)
            
            if self.config.show_grid:
                ax.grid(True, color=self.config.grid_color, zorder=0)
            
            # Rotate and scale labels for readability
            plt.xticks(rotation=60, ha='right', fontsize=8)
            plt.tight_layout()
            
            # Save in each format
            for fmt in formats:
                output_dir = Path(self.config.output_dir) / fmt
                output_dir.mkdir(parents=True, exist_ok=True)
                filepath = output_dir / f"{safe_name}.{fmt}"
                fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                generated_files.append(str(filepath))
            
            plt.close(fig)
        
        return generated_files
    
    def _generate_chart_svg(self, group: pd.DataFrame, name: str) -> str:
        """
        Generate a chart for a position group and return it as an inline SVG string.
        
        Args:
            group: DataFrame group for a single position
            name: Position name (used as chart title)
            
        Returns:
            Inline SVG string (the <svg>...</svg> element, no XML declaration)
        """
        group = group.sort_values(by=self.SAL_MAX, ascending=True)
        
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        
        heights = group[self.SAL_MAX] - group[self.SAL_MIN]
        linewidths = [3 if h == 0 else 1 for h in heights]
        
        bars = ax.bar(
            group[self.LOCATION],
            heights,
            bottom=group[self.SAL_MIN],
            color=group[self.COLOR],
            edgecolor=self.config.edge_color,
            linewidth=linewidths,
            zorder=3,
            width=self.config.bar_width
        )
        
        if self.config.show_labels:
            for bar, (_, row) in zip(bars, group.iterrows()):
                height = bar.get_height()
                ax.annotate(
                    f'${row[self.SAL_MAX]:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8
                )
        
        ax.set_ylabel(self.config.ylabel)
        ax.set_xlabel(self.config.xlabel)
        ax.set_title(name)
        
        if self.config.show_grid:
            ax.grid(True, color=self.config.grid_color, zorder=0)
        
        plt.xticks(rotation=60, ha='right', fontsize=8)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        svg_string = buf.read().decode('utf-8')
        # Strip XML declaration, keep just the <svg>...</svg> element
        return svg_string[svg_string.find('<svg'):]

    def _generate_html_report(
        self,
        df: pd.DataFrame,
        client_name: str,
        input_file: str
    ) -> str:
        """Generate comprehensive HTML report with embedded charts"""
        logger.info("Generating HTML report")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        position_summaries = []
        
        for position_name, group in df.groupby(self.TITLE):
            safe_name = position_name.replace('/', '_')
            sorted_group = group.sort_values(by=self.SAL_MAX, ascending=True)
            
            employers_data = []
            for _, row in sorted_group.iterrows():
                employers_data.append({
                    'employer': row[self.LOCATION],
                    'min_salary': row[self.SAL_MIN],
                    'max_salary': row[self.SAL_MAX],
                    'is_client': client_name in row[self.LOCATION]
                })
            
            chart_svg = self._generate_chart_svg(group, position_name)
            
            stats = self.stats.get(position_name)
            position_summaries.append({
                'name': position_name,
                'safe_name': safe_name,
                'employers': employers_data,
                'chart_svg': chart_svg,
                'stats': stats
            })
        
        html_content = self._build_html_content(
            position_summaries, df, client_name, timestamp, input_file
        )
        
        # Save HTML file
        output_dir = Path(self.config.output_dir) / 'html'
        output_dir.mkdir(parents=True, exist_ok=True)
        html_filename = f"compensation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_path = output_dir / html_filename
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {html_path}")
        return str(html_path)
    
    def _build_html_content(
        self,
        position_summaries: List[Dict],
        df: pd.DataFrame,
        client_name: str,
        timestamp: str,
        input_file: str
    ) -> str:
        """Build the HTML content string"""
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compensation Analysis Report - {client_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .position-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .position-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .salary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .salary-table th, .salary-table td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        .salary-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .client-row {{
            background-color: #e8f4fd;
            font-weight: bold;
        }}
        .salary-range {{
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }}
        .summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-box {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .position-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .position-stat {{
            text-align: center;
        }}
        .position-stat-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .position-stat-label {{
            font-size: 0.8em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Compensation Analysis Report</h1>
        <h2>Client: {client_name}</h2>
        <p>Generated on {timestamp}</p>
        <p>Data source: {os.path.basename(input_file)}</p>
    </div>

    <div class="summary">
        <h2>Report Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{len(position_summaries)}</div>
                <div class="stat-label">Positions Analyzed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(set(emp['employer'] for pos in position_summaries for emp in pos['employers']))}</div>
                <div class="stat-label">Employers Compared</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${df[self.SAL_MIN].min():,.0f} - ${df[self.SAL_MAX].max():,.0f}</div>
                <div class="stat-label">Salary Range</div>
            </div>
        </div>
    </div>

    <h2>Position Details</h2>
'''

        for position in position_summaries:
            stats = position.get('stats')
            stats_html = ''
            if stats:
                stats_html = f'''
        <div class="position-stats">
            <div class="position-stat">
                <div class="position-stat-value">{stats.employer_count}</div>
                <div class="position-stat-label">Employers</div>
            </div>
            <div class="position-stat">
                <div class="position-stat-value">${stats.min_salary:,.0f}</div>
                <div class="position-stat-label">Min Salary</div>
            </div>
            <div class="position-stat">
                <div class="position-stat-value">${stats.max_salary:,.0f}</div>
                <div class="position-stat-label">Max Salary</div>
            </div>
            <div class="position-stat">
                <div class="position-stat-value">${stats.median_salary:,.0f}</div>
                <div class="position-stat-label">Median</div>
            </div>
            <div class="position-stat">
                <div class="position-stat-value">${stats.mean_salary:,.0f}</div>
                <div class="position-stat-label">Mean</div>
            </div>
        </div>
'''

            html_content += f'''
    <div class="position-card">
        <div class="position-title">{position['name']}</div>
        {stats_html}
        <div class="chart-container">
            {position['chart_svg']}
        </div>

        <table class="salary-table">
            <thead>
                <tr>
                    <th>Employer</th>
                    <th>Minimum Salary</th>
                    <th>Maximum Salary</th>
                    <th>Salary Range</th>
                </tr>
            </thead>
            <tbody>
'''

            for employer in position['employers']:
                row_class = 'client-row' if employer['is_client'] else ''
                html_content += f'''
                <tr class="{row_class}">
                    <td>{employer['employer']}</td>
                    <td>${employer['min_salary']:,.0f}</td>
                    <td>${employer['max_salary']:,.0f}</td>
                    <td class="salary-range">${employer['min_salary']:,.0f} - ${employer['max_salary']:,.0f}</td>
                </tr>
'''

            html_content += '''
            </tbody>
        </table>
    </div>
'''

        html_content += '''
</body>
</html>
'''
        return html_content
    
    def generate_summary_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a text summary report of all statistics.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Summary report as string
        """
        if not self.stats:
            return "No statistics available. Run calculate_statistics() first."
        
        lines = [
            "=" * 60,
            "COMPENSATION ANALYSIS SUMMARY REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Positions Analyzed: {len(self.stats)}",
            "-" * 60,
            ""
        ]
        
        for position, stats in sorted(self.stats.items()):
            lines.extend([
                f"Position: {position}",
                f"  Employers:     {stats.employer_count}",
                f"  Min Salary:    ${stats.min_salary:,.2f}",
                f"  Max Salary:    ${stats.max_salary:,.2f}",
                f"  Median:        ${stats.median_salary:,.2f}",
                f"  Mean:          ${stats.mean_salary:,.2f}",
                f"  Range:         ${stats.salary_range:,.2f}",
                ""
            ])
        
        lines.append("=" * 60)
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Summary report saved to: {output_path}")
        
        return report


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create custom config
    config = GraphConfig(
        client_color='#4CAF50',
        show_grid=True,
        show_labels=True
    )
    
    generator = GraphGenerator(config)
    print("GraphGenerator module loaded successfully")
