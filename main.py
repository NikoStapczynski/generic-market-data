import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging


class FriendlyArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that prints helpful error messages"""
    
    def error(self, message):
        print("\n" + "="*60)
        print("ERROR: Invalid command usage")
        print("="*60)
        print(f"\n{message}\n")
        print("USAGE:")
        print("  python main.py [OPTIONS]\n")
        print("REQUIRED:")
        print("  -i FILE        Path to data file (.csv, .xls, .xlsx, .ods)")
        print("                      Default: input/csv/example_table.csv\n")
        print("OPTIONAL:")
        print("  --client NAME       Name of client to highlight (auto-detected if not provided)")
        print("  --output FORMATS    Output format(s): html, pdf, png, svg, jpg, jpeg, webp, eps")
        print("                      Default: png")
        print("  -h, --help          Show full help message\n")
        print("EXAMPLES:")
        print("  python main.py")
        print("  python main.py -i data.csv")
        print("  python main.py -i data.xlsx --client 'Company Name'")
        print("  python main.py -i data.csv --output html png svg")
        print("="*60 + "\n")
        sys.exit(2)

salary = 'salary'
location = 'location'
sal_min = 'salary_min'
sal_max = 'salary_max'
title = 'POSITION TITLE'

def read_data(file_path, ext):
    converters = {title: str} if title else {}
    if ext == '.csv':
        return pd.read_csv(file_path, converters=converters)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path, converters=converters)
    elif ext == '.ods':
        return pd.read_excel(file_path, engine='odf', converters=converters)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def remove_summary_columns(df):
    bad_columns = [
        "Comp Data Points",
        "Comp Average",
        "Comp Lo-Hi Range",
        "Comp Median",
        "75th Percent of Market",
        "% Melrose Higher Lower than 75th Percentile"
    ]
    for c in bad_columns:
        df = df.drop(c, axis=1, errors='ignore')

    return df


def combine_lines(df):
    newdf = pd.DataFrame()
    every_other_idx = df.index // 2
    for column in df.columns:
        newdf[column] = df.groupby(every_other_idx)[column].apply(list)

    newdf[title] = newdf.apply(lambda row: [row[title][0]] * 2, axis=1)
    return newdf


def normalize(df):
    first = pd.DataFrame()
    second = pd.DataFrame()
    for c in df.columns:
        first[c] = df.apply(lambda row: row[c][0], axis=1)
        second[c] = df.apply(lambda row: row[c][1], axis=1)

    return pd.concat([first, second])


def make_city_column(df):
    df = df.melt(
        id_vars=[title],
        value_vars=list(df.columns[1:]),
        value_name=salary,
        var_name=location
    ).dropna()
    # Convert salary to numeric
    df[salary] = pd.to_numeric(df[salary].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce')
    df = df.dropna(subset=[salary])
    # Filter out empty titles
    df = df[df[title].str.len() > 0]
    return df


def combine_high_low(df, location_name):
    groups = df.groupby([location, title])

    def helper(group):
        minimum = group[salary].min()
        maximum = group[salary].max()

        if minimum == maximum:
            minimum = minimum * .99
            maximum = maximum

        client_color = '#AAA'
        default_color = '#FFF'

        if location_name in group[location].iloc[0]:
            color = client_color
        else:
            color = default_color

        return pd.Series(
            data={
                location: group[location].iloc[0],
                title: group[title].iloc[0],
                sal_min: minimum,
                sal_max: maximum,
                'color': color
            },
            index=[location, title, sal_min, sal_max, 'color']
        )

    return pd.concat([helper(group) for name, group in groups], axis=1).transpose()


def graph(df, output):

    for name, group in df.groupby(title):
        # Sanitize name for filename by replacing slashes with underscores
        safe_name = name.replace('/', '_')

        group = group.sort_values(by=sal_max, ascending=True)
        fig, ax = plt.subplots()
        heights = group[sal_max] - group[sal_min]
        linewidths = [3 if h == 0 else 1 for h in heights]
        ax.bar(group[location], heights, bottom=group[sal_min], color=group['color'], edgecolor='black', linewidth=linewidths, zorder=3)

        ax.set_ylabel("Pay Range (Hourly)")
        ax.set_xlabel("Location")
        ax.set_title(name)
        ax.grid(True, color="#AAA", zorder=0)

        output_configs = {
            'pdf': ('output/pdf', 'pdf'),
            'png': ('output/png', 'png'),
            'svg': ('output/svg', 'svg'),
            'jpg': ('output/jpg', 'jpg'),
            'jpeg': ('output/jpeg', 'jpeg'),
            'webp': ('output/webp', 'webp'),
            'eps': ('output/eps', 'eps'),
        }
        for fmt in output:
            if fmt in output_configs:
                dir_path, ext = output_configs[fmt]
                os.makedirs(dir_path, exist_ok=True)
                fig.savefig(f"{dir_path}/{safe_name}.{fmt}")
        plt.close(fig)


def graph_with_html(df, output_formats, client_name, input_file):
    """Generate graphs with optional HTML output"""
    from datetime import datetime

    # Generate image formats first
    image_formats = [fmt for fmt in output_formats if fmt != 'html']
    if image_formats:
        graph(df, image_formats)

    # Generate HTML if requested
    if 'html' in output_formats:
        generate_html_report(df, client_name, input_file)


def generate_html_report(df, client_name, input_file):
    """Generate HTML report with embedded charts"""
    from datetime import datetime

    # Group data by position for HTML generation
    position_summaries = []

    for position_name, group in df.groupby(title):
        safe_name = position_name.replace('/', '_')

        # Sort by salary range
        group = group.sort_values(by=sal_max, ascending=True)

        # Create position summary
        employers_data = []
        for _, row in group.iterrows():
            employers_data.append({
                'employer': row[location],
                'min_salary': row[sal_min],
                'max_salary': row[sal_max],
                'is_client': client_name in row[location]
            })

        position_summaries.append({
            'name': position_name,
            'safe_name': safe_name,
            'employers': employers_data,
            'chart_file': f'../png/{safe_name}.png'
        })

    # Generate HTML
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
                <div class="stat-value">${df[sal_min].min():,.0f} - ${df[sal_max].max():,.0f}</div>
                <div class="stat-label">Salary Range</div>
            </div>
        </div>
    </div>

    <h2>Position Details</h2>
'''

    for position in position_summaries:
        html_content += f'''
    <div class="position-card">
        <div class="position-title">{position['name']}</div>

        <div class="chart-container">
            <img src="{position['chart_file']}" alt="Salary comparison for {position['name']}" style="max-width: 100%; height: auto;">
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

    # Save HTML file
    os.makedirs('output/html', exist_ok=True)
    html_filename = f"compensation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    html_path = os.path.join('output/html', html_filename)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML report saved to: {html_path}")


def print_error(message, suggestion=None):
    """Print a formatted error message with optional suggestion"""
    print("\n" + "="*60)
    print("ERROR")
    print("="*60)
    print(f"\n{message}")
    if suggestion:
        print(f"\nSuggestion: {suggestion}")
    print("="*60 + "\n")


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = FriendlyArgumentParser(description='Generate floating bar graphs for compensation data.')
    parser.add_argument('--client', type=str, help='Name of the client to be highlighted. Defaults to the first employer found in the data set', metavar='Employer')
    parser.add_argument('-i', type=str, default='input/csv/example_table.csv', help='Path to data file (supports .csv, .xls, .xlsx, .ods)', metavar='path/to/file')
    parser.add_argument('--output', nargs='+', default=['png'], choices=['html', 'pdf', 'png', 'svg', 'jpg', 'jpeg', 'webp', 'eps'], help='Output formats: html, pdf, png, svg, jpg, jpeg, webp, eps', metavar='file extension')

    args = parser.parse_args()

    file_path = args.i
    
    # Check if file exists
    if not os.path.exists(file_path):
        print_error(
            f"File not found: {file_path}",
            "Check that the file path is correct. Example: -i input/csv/mydata.csv"
        )
        sys.exit(1)
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.csv', '.xls', '.xlsx', '.ods']:
        print_error(
            f"Unsupported file format: {ext}",
            "Supported formats: .csv, .xls, .xlsx, .ods"
        )
        sys.exit(1)

    # Read columns to find title and possibly client
    try:
        if ext == '.csv':
            columns = pd.read_csv(file_path, nrows=0).columns.tolist()
        elif ext in ['.xls', '.xlsx']:
            columns = pd.read_excel(file_path, nrows=0).columns.tolist()
        elif ext == '.ods':
            columns = pd.read_excel(file_path, engine='odf', nrows=0).columns.tolist()
    except Exception as e:
        print_error(
            f"Could not read file: {e}",
            "Make sure the file is not corrupted and has the correct format"
        )
        sys.exit(1)

    # Find the title column (contains 'POSITION TITLE')
    title = None
    title_col_index = None
    for i, col in enumerate(columns):
        if 'POSITION TITLE' in col.upper():
            title = col
            title_col_index = i
            break

    if title is None:
        print_error(
            "Could not find 'POSITION TITLE' column in the file",
            "Make sure your data has a column with 'POSITION TITLE' in its header"
        )
        sys.exit(1)

    # If client not provided, set from CSV
    if args.client is None:
        if title_col_index is not None and title_col_index + 1 < len(columns):
            client_col = columns[title_col_index + 1]
            # Parse the client name: take part before 'Current' if present
            if 'Current' in client_col:
                args.client = client_col.split('Current')[0].strip()
            else:
                args.client = client_col
        else:
            print_error(
                "Client name not provided and could not determine from file",
                "Use --client 'Client Name' to specify the client to highlight"
            )
            sys.exit(1)

    try:
        df = read_data(file_path, ext)
    except Exception as e:
        print_error(
            f"Error reading data file: {e}",
            "Make sure the file format matches the extension"
        )
        sys.exit(1)
        
    df = remove_summary_columns(df)

    # Check if job titles are on every line (no empty titles)
    titles_present = df[title].notna() & (df[title].str.strip() != '')
    if titles_present.all():
        # Titles on every line: no pairing needed
        pass  # Proceed directly to make_city_column
    else:
        # Titles on every other line: use pairing logic
        # Ensure even number of rows
        if len(df) % 2 == 1:
            df = df.iloc[:-1]
        df = combine_lines(df)
        df = normalize(df)

    df = make_city_column(df)
    df = combine_high_low(df, args.client)
    graph_with_html(df, args.output, args.client, args.i)
    
    print("\n" + "="*60)
    print("Success! Charts generated.")
    print("="*60)
    abs_output_path = os.path.abspath('output')
    abs_input_path = os.path.abspath(file_path)
    print(f"\nOutput location:\t{abs_output_path}/")
    print(f"Input file:\t\t{abs_input_path}")
    print(f"Client highlighted:\t{args.client}")
    print(f"Output formats:\t\t{', '.join(args.output)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
