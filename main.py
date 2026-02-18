import os
import re
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

# Keywords that indicate a job is paid per-inspection rather than hourly
PER_INSPECTION_INDICATORS = ['per', 'inspection', 'fee', 'paid']


def is_per_inspection(value) -> bool:
    """Return True if a cell value signals per-inspection (not hourly) pay."""
    if value is None:
        return False
    try:
        import math
        if math.isnan(float(value)):
            return False
    except (ValueError, TypeError):
        pass
    s = str(value).strip().lower()
    return any(indicator in s for indicator in PER_INSPECTION_INDICATORS)

def format_per_inspection_rate(rates):
    """Format a list of numeric inspection rates into a display string."""
    if not rates:
        return "Per Inspection"
    unique = sorted(set(rates))
    if len(unique) == 1:
        return f"${unique[0]:,.2f} per inspection"
    return f"${min(unique):,.2f} \u2013 ${max(unique):,.2f} per inspection"


def read_data(file_path, ext, header_row=0):
    converters = {title: str} if title else {}
    if ext == '.csv':
        return pd.read_csv(file_path, converters=converters, skiprows=header_row, header=0)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path, converters=converters, header=header_row)
    elif ext == '.ods':
        return pd.read_excel(file_path, engine='odf', converters=converters, header=header_row)
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
    melted = df.melt(
        id_vars=[title],
        value_vars=list(df.columns[1:]),
        value_name=salary,
        var_name=location
    ).dropna()

    # Detect per-inspection rows BEFORE numeric coercion, then drop them from
    # the hourly dataset so they never appear as a numeric bar in the chart.
    per_inspection_raw = {}  # {position: {employer: [raw_strings]}}
    per_inspection_indices = []
    for idx, row in melted.iterrows():
        if is_per_inspection(row[salary]):
            pos = row[title]
            emp = row[location]
            per_inspection_raw.setdefault(pos, {}).setdefault(emp, []).append(
                str(row[salary]).strip()
            )
            per_inspection_indices.append(idx)

    # Remove per-inspection rows from the hourly data entirely
    melted = melted.drop(index=per_inspection_indices)

    # Build per_inspection dict: extract numeric rates from raw strings
    per_inspection = {}
    for pos, employers in per_inspection_raw.items():
        per_inspection[pos] = {}
        for emp, raw_values in employers.items():
            rates = []
            for v in raw_values:
                nums = re.findall(r'\d+\.?\d*', v)
                rates.extend(float(n) for n in nums)
            per_inspection[pos][emp] = sorted(set(rates))

    # Convert salary to numeric
    melted[salary] = pd.to_numeric(
        melted[salary].astype(str).str.extract(r'(\d+\.?\d*)', expand=False),
        errors='coerce'
    )
    melted = melted.dropna(subset=[salary])
    # Filter out empty titles
    melted = melted[melted[title].str.len() > 0]
    return melted, per_inspection


def combine_high_low(df, location_name):
    groups = df.groupby([location, title])

    def helper(group):
        minimum = group[salary].min()
        maximum = group[salary].max()

        # if minimum == maximum:
        #     minimum = minimum * .99
        #     maximum = maximum

        client_color = '#e8f4fd'#'#AAA'
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

        ax.set_ylabel("Hourly Pay")
        ax.set_xlabel("Location")
        # ax.set_title(name)
        ax.grid(True, color="#AAA", zorder=0)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        plt.setp(ax.get_xticklabels(), ha="right")
        plt.tight_layout()

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
                fig.savefig(f"{dir_path}/{safe_name}.{fmt}", bbox_inches="tight")
        plt.close(fig)


def chart_to_svg(group, name):
    """Generate a bar chart for a position group and return it as an inline SVG string."""
    import io as _io

    group = group.sort_values(by=sal_max, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    heights = group[sal_max] - group[sal_min]
    linewidths = [3 if h == 0 else 1 for h in heights]
    ax.bar(group[location], heights, bottom=group[sal_min], color=group['color'],
           edgecolor='black', linewidth=linewidths, zorder=3)
    ax.set_ylabel("Hourly Pay")
    ax.set_xlabel("Location")
    #ax.set_title(name)
    ax.grid(True, color="#AAA", zorder=0)
    plt.xticks(rotation=60, ha='right', fontsize=8)
    plt.tight_layout()

    buf = _io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    svg_string = buf.read().decode('utf-8')
    # Strip XML declaration, keep just the <svg>...</svg> element
    svg_content = svg_string[svg_string.find('<svg'):]
    return svg_content


def graph_with_html(df, output_formats, client_name, input_file, per_inspection=None):
    """Generate graphs with optional HTML output"""
    # Generate image formats (non-HTML)
    image_formats = [fmt for fmt in output_formats if fmt != "html"]
    if image_formats:
        graph(df, image_formats)

    # Generate HTML if requested
    if 'html' in output_formats:
        generate_html_report(df, client_name, input_file, per_inspection or {})


def generate_html_report(df, client_name, input_file, per_inspection=None):
    """Generate a self-contained HTML report with charts embedded as base64 data URIs."""
    from datetime import datetime

    if per_inspection is None:
        per_inspection = {}

    # Group data by position for HTML generation
    position_summaries = []

    for position_name, group in df.groupby(title):
        safe_name = position_name.replace('/', '_')

        # Sort by salary range
        sorted_group = group.sort_values(by=sal_max, ascending=True)

        # Create position summary (hourly employers)
        employers_data = []
        for _, row in sorted_group.iterrows():
            employers_data.append({
                'employer': row[location],
                'min_salary': row[sal_min],
                'max_salary': row[sal_max],
                'is_client': client_name in row[location],
                'per_inspection': False,
            })

        # Append per-inspection employers for this position (sorted by name)
        for pi_employer, rates in sorted(per_inspection.get(position_name, {}).items()):
            employers_data.append({
                'employer': pi_employer,
                'is_client': client_name in pi_employer,
                'per_inspection': True,
                'rates': rates,
            })

        chart_svg = chart_to_svg(group, position_name)

        position_summaries.append({
            'name': position_name,
            'safe_name': safe_name,
            'employers': employers_data,
            'chart_svg': chart_svg
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
            background: linear-gradient(135deg, #667eea 0%, #aeceaf 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        /* ── Legend / Table of Contents ── */
        .legend {{
            background: white;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .legend h2 {{
            margin: 0 0 12px 0;
            font-size: 1.1em;
            color: #2c3e50;
        }}
        .search-bar {{
            width: 100%;
            box-sizing: border-box;
            padding: 8px 12px;
            font-size: 0.95em;
            border: 1px solid #ccc;
            border-radius: 6px;
            margin-bottom: 14px;
            outline: none;
            transition: border-color 0.2s;
        }}
        .search-bar:focus {{
            border-color: #3498db;
        }}
        .legend-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px 10px;
            list-style: none;
            margin: 0;
            padding: 0;
            max-height: 220px;
            overflow-y: auto;
        }}
        .legend-list li {{
            flex: 0 0 auto;
        }}
        .legend-list a {{
            display: inline-block;
            padding: 3px 10px;
            background: #eaf4fb;
            border: 1px solid #b6d9f0;
            border-radius: 20px;
            color: #2471a3;
            text-decoration: none;
            font-size: 0.85em;
            transition: background 0.15s, color 0.15s;
            white-space: nowrap;
        }}
        .legend-list a:hover {{
            background: #3498db;
            color: white;
            border-color: #2980b9;
        }}
        .legend-item-hidden {{
            display: none !important;
        }}
        .no-results {{
            color: #999;
            font-style: italic;
            font-size: 0.9em;
            padding: 4px 0;
        }}
        /* ── Position Cards ── */
        .position-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            scroll-margin-top: 20px;
        }}
        .position-card.card-hidden {{
            display: none;
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
        .per-inspection-row td {{
            color: #888;
            font-style: italic;
        }}
        .per-inspection-badge {{
            display: inline-block;
            background: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            border-radius: 4px;
            padding: 1px 7px;
            font-size: 0.82em;
            font-style: normal;
            white-space: nowrap;
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
        /* ── Back-to-top button ── */
        #backToTop {{
            display: none;
            position: fixed;
            bottom: 28px;
            right: 28px;
            z-index: 999;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 50px;
            padding: 10px 16px;
            font-size: 0.9em;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 3px 10px rgba(0,0,0,0.25);
            transition: background 0.2s, opacity 0.2s;
            opacity: 0.85;
        }}
        #backToTop:hover {{
            background: #2980b9;
            opacity: 1;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Compensation Analysis Report</h1>
        <h2>Client: {client_name}</h2>
    </div>

    <!-- Legend / Table of Contents -->
    <div class="legend" id="legend">
        <h2>&#128269; Positions ({len(position_summaries)} total)</h2>
        <input
            class="search-bar"
            id="positionSearch"
            type="search"
            placeholder="Search positions&hellip;"
            autocomplete="off"
        />
        <ul class="legend-list" id="legendList">
'''

    for position in position_summaries:
        html_content += f'            <li class="legend-item" data-name="{position["name"].lower()}"><a href="#pos-{position["safe_name"]}">{position["name"]}</a></li>\n'

    html_content += '''        </ul>
        <p class="no-results" id="noResults" style="display:none;">No positions match your search.</p>
    </div>

'''

    for position in position_summaries:
        html_content += f'''
    <div class="position-card" id="pos-{position['safe_name']}" data-position-name="{position['name'].lower()}">
        <div class="position-title">{position['name']}</div>

        <div class="chart-container">
            {position['chart_svg']}
        </div>

        <table class="salary-table">
            <thead>
                <tr>
                    <th>Employer</th>
                    <th>Minimum</th>
                    <th>Maximum</th>
                </tr>
            </thead>
            <tbody>
'''

        for employer in position['employers']:
            if employer['per_inspection']:
                rate_str = format_per_inspection_rate(employer['rates'])
                row_class = 'per-inspection-row' + (' client-row' if employer['is_client'] else '')
                html_content += f'''
                <tr class="{row_class}">
                    <td>{employer['employer']}</td>
                    <td colspan="2"><span class="per-inspection-badge">&#128338; {rate_str}</span></td>
                </tr>
'''
            else:
                row_class = 'client-row' if employer['is_client'] else ''
                html_content += f'''
                <tr class="{row_class}">
                    <td>{employer['employer']}</td>
                    <td>${employer['min_salary']:,.2f}</td>
                    <td>${employer['max_salary']:,.2f}</td>
                </tr>
'''

        html_content += '''
            </tbody>
        </table>
    </div>
'''

    html_content += '''
    <button id="backToTop" title="Back to top">&#8679; Top</button>

    <script>
    (function () {
        const searchInput = document.getElementById('positionSearch');
        const legendItems = document.querySelectorAll('.legend-item');
        const cards = document.querySelectorAll('.position-card');
        const noResults = document.getElementById('noResults');
        const backToTop = document.getElementById('backToTop');

        // Search filtering
        searchInput.addEventListener('input', function () {
            const query = this.value.trim().toLowerCase();
            let visibleCount = 0;

            legendItems.forEach(function (li) {
                const name = li.getAttribute('data-name');
                if (!query || name.includes(query)) {
                    li.classList.remove('legend-item-hidden');
                    visibleCount++;
                } else {
                    li.classList.add('legend-item-hidden');
                }
            });

            cards.forEach(function (card) {
                const name = card.getAttribute('data-position-name');
                if (!query || name.includes(query)) {
                    card.classList.remove('card-hidden');
                } else {
                    card.classList.add('card-hidden');
                }
            });

            noResults.style.display = visibleCount === 0 ? 'block' : 'none';
        });

        // Show/hide back-to-top button
        window.addEventListener('scroll', function () {
            backToTop.style.display = window.scrollY > 300 ? 'block' : 'none';
        });

        // Back-to-top click: scroll to top and clear the hash
        backToTop.addEventListener('click', function () {
            window.scrollTo({ top: 0, behavior: 'smooth' });
            history.replaceState(null, '', window.location.pathname + window.location.search);
        });
    })();
    </script>
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
            # Check if first row has meaningful headers or if we need to skip
            first_line = pd.read_csv(file_path, nrows=1).columns.tolist()
            if all('Unnamed' in col or col.strip() == '' for col in first_line):
                # Skip first row, headers are on row 2
                columns = pd.read_csv(file_path, skiprows=1, nrows=0).columns.tolist()
                header_row = 1
            else:
                columns = first_line
                header_row = 0
        elif ext in ['.xls', '.xlsx']:
            columns = pd.read_excel(file_path, nrows=0).columns.tolist()
            header_row = 0
        elif ext == '.ods':
            columns = pd.read_excel(file_path, engine='odf', nrows=0).columns.tolist()
            header_row = 0
    except Exception as e:
        print_error(
            f"Could not read file: {e}",
            "Make sure the file is not corrupted and has the correct format"
        )
        sys.exit(1)

    # Find the title column (contains 'POSITION TITLE' or 'DARTMOUTH POSITION TITLES')
    title = None
    title_col_index = None
    for i, col in enumerate(columns):
        if 'POSITION TITLE' in col.upper() or 'DARTMOUTH POSITION' in col.upper():
            title = col
            title_col_index = i
            break

    if title is None:
        print_error(
            "Could not find position title column in the file",
            "Make sure your data has a column with 'POSITION TITLE' or 'DARTMOUTH POSITION TITLES' in its header"
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
        df = read_data(file_path, ext, header_row)
    except Exception as e:
        print_error(
            f"Error reading data file: {e}",
            "Make sure the file format matches the extension"
        )
        sys.exit(1)
        
    # Rename the detected title column to the standard 'POSITION TITLE' for consistency
    if title != 'POSITION TITLE':
        df = df.rename(columns={title: 'POSITION TITLE'})
        title = 'POSITION TITLE'
        
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

    df, per_inspection = make_city_column(df)
    df = combine_high_low(df, args.client)
    graph_with_html(df, args.output, args.client, args.i, per_inspection)
    
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
