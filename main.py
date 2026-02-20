import io
import math
import os
import re
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class FriendlyArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that prints helpful error messages"""
    
    def error(self, message):
        print("\n" + "="*60)
        print("ERROR: Invalid command usage")
        print("="*60)
        print(f"\n{message}\n")
        print("USAGE:")
        print("  python main.py [OPTIONS] [FORMAT ...]\n")
        print("REQUIRED:")
        print("  -i FILE        Path to data file (.csv, .xls, .xlsx, .ods)")
        print("                      Default: input/csv/example_table.csv\n")
        print("OPTIONAL:")
        print("  --client NAME  Name of client to highlight (auto-detected if not provided)")
        print("  FORMAT ...     Output format(s): html, pdf, png, svg, jpg, jpeg, webp, eps")
        print("                      Default: png")
        print("  -h, --help     Show full help message\n")
        print("EXAMPLES:")
        print("  python main.py")
        print("  python main.py -i data.csv")
        print("  python main.py -i data.xlsx --client 'Company Name'")
        print("  python main.py -i data.csv html png svg")
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


def classify_special_status(combined_text: str) -> tuple:
    """
    Given a combined string assembled from all non-numeric cell values for a
    single (position, employer) pair, determine the special status type.

    Returns a 3-tuple ``(status_type, display_text, reference)`` where:

    * ``status_type`` – one of ``'district'``, ``'outsourced'``, ``'see'``,
      or ``None`` (unrecognised / skip).
    * ``display_text`` – human-readable label shown in the badge.
    * ``reference``    – for ``'see'`` type, the raw reference string used to
      look up a matching position; ``None`` otherwise.
    """
    # Normalise whitespace
    t = ' '.join(combined_text.split()).strip()
    tl = t.lower()

    if not tl:
        return (None, t, None)

    if 'outsourced' in tl:
        return ('outsourced', 'Outsourced', None)

    # "Cape Cod District" may appear split across two rows as "District" and
    # "Cape Cod", so accept any combination that mentions both or either.
    if 'cape cod' in tl or 'district' in tl:
        return ('district', 'Cape Cod District', None)

    if tl.startswith('see'):
        ref = t[3:].strip()   # everything after "see"
        return ('see', ref, ref)

    if 'done by' in tl:
        ref = re.sub(r'(?i)done\s+by\s*', '', t).strip()
        return ('done_by', ref, ref if ref else None)

    return (None, t, None)


def find_position_match(reference: str, position_names: List[str]) -> Optional[str]:
    """
    Try to find a position name in *position_names* that best matches the
    short *reference* string (e.g. ``"Fin Dir"``).

    Strategy
    --------
    1. Direct substring match (reference inside position name).
    2. Word-prefix overlap: count how many reference words are a prefix of any
       word in the position name.  Require ≥ 50 % of reference words to match.
    """
    if not reference:
        return None
    ref_lower = reference.lower()
    ref_words = ref_lower.split()
    if not ref_words:
        return None

    # 1. substring match
    for pos in position_names:
        if ref_lower in pos.lower():
            return pos

    # 2. word-prefix overlap
    best_match = None
    best_score = 0
    for pos in position_names:
        pos_words = pos.lower().replace('/', ' ').replace('&', ' ').split()
        score = sum(
            1 for rw in ref_words
            if any(pw.startswith(rw) or rw.startswith(pw) for pw in pos_words)
        )
        if score > best_score:
            best_score = score
            best_match = pos

    threshold = max(1, len(ref_words) * 0.5)
    return best_match if best_score >= threshold else None


def render_special_status_badge(ss_info: dict, position_summaries: list) -> str:
    """
    Return an HTML string (a styled ``<span>`` badge) for the given special
    status.  For ``'see'`` type, attempt to link to the referenced position.
    """
    status_type = ss_info['type']
    display     = ss_info['display']
    reference   = ss_info.get('reference')

    if status_type == 'district':
        return (
            '<span class="status-badge badge-district">'
            '&#127963; Cape Cod District'
            '</span>'
        )

    if status_type == 'outsourced':
        return (
            '<span class="status-badge badge-outsourced">'
            '&#128260; Outsourced'
            '</span>'
        )

    if status_type == 'see':
        if reference and reference.lower() != 'above':
            matched = find_position_match(
                reference, [p['name'] for p in position_summaries]
            )
            if matched:
                safe = matched.replace('/', '_')
                return (
                    f'<span class="status-badge badge-see">'
                    f'&#128279; See: <a href="#pos-{safe}">{matched}</a>'
                    f'</span>'
                )
        # Fall-through: show plain text (includes "see above")
        label = f'See: {display}' if display else 'See above'
        return f'<span class="status-badge badge-see">&#128279; {label}</span>'

    if status_type == 'done_by':
        if reference:
            matched = find_position_match(
                reference, [p['name'] for p in position_summaries]
            )
            if matched:
                safe = matched.replace('/', '_')
                return (
                    f'<span class="status-badge badge-done-by">'
                    f'&#128101; Done by: <a href="#pos-{safe}">{matched}</a>'
                    f'</span>'
                )
        label = f'Done by: {display}' if display else 'Done by another position'
        return f'<span class="status-badge badge-done-by">&#128101; {label}</span>'

    # Unknown / unclassified
    return f'<span class="status-badge badge-unknown">&#8505; {display}</span>'


def read_data(file_path: str, ext: str, header_row: int = 0) -> pd.DataFrame:
    """Read a compensation data file into a DataFrame."""
    converters = {title: str}
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

    # Build per_inspection dict: extract numeric rates from raw strings.
    # Also detect "fee paid" entries (cells containing "fee"/"paid" but not
    # "per"/"inspection") so they can be labelled differently in the output.
    per_inspection = {}
    for pos, employers in per_inspection_raw.items():
        per_inspection[pos] = {}
        for emp, raw_values in employers.items():
            rates = []
            for v in raw_values:
                nums = re.findall(r'\d+\.?\d*', v)
                rates.extend(float(n) for n in nums)
            combined_raw = ' '.join(raw_values).lower()
            has_per_or_inspection = 'per' in combined_raw or 'inspection' in combined_raw
            has_fee_or_paid = 'fee' in combined_raw or 'paid' in combined_raw
            is_fee_paid = has_fee_or_paid and not has_per_or_inspection
            per_inspection[pos][emp] = {
                'rates': sorted(set(rates)),
                'fee_paid': is_fee_paid,
            }

    # ------------------------------------------------------------------ #
    # Detect other special statuses: Cape Cod District, outsourced,       #
    # "see <position>" references, etc.                                   #
    # These are non-numeric cell values that are NOT per-inspection.      #
    # They are collected here and removed before the numeric conversion   #
    # so they never produce NaN bars in the chart.                        #
    # ------------------------------------------------------------------ #
    per_insp_idx_set = set(per_inspection_indices)
    special_raw: Dict[str, Dict[str, List[str]]] = {}  # {pos: {emp: [values]}}
    special_status_indices = []

    for idx, row in melted.iterrows():
        if idx in per_insp_idx_set:
            continue  # already handled
        val = str(row[salary]).strip()
        if not val:
            continue
        # Skip values that are purely numeric (with optional $, commas, etc.)
        try:
            float(val.replace(',', '').replace('$', ''))
            continue  # numeric — leave for the normal conversion
        except (ValueError, TypeError):
            pass
        # Also skip values that look numeric via the extraction regex
        if re.fullmatch(r'\$?\s*[\d,]+\.?\d*', val):
            continue
        pos = row[title]
        emp = row[location]
        special_raw.setdefault(pos, {}).setdefault(emp, []).append(val)
        special_status_indices.append(idx)

    # Drop special-status rows so they don't end up as NaN after coercion
    melted = melted.drop(index=special_status_indices)

    # Classify the combined text for each (position, employer) pair
    special_statuses: Dict[str, Dict[str, dict]] = {}
    for pos, employers in special_raw.items():
        for emp, values in employers.items():
            combined = ' '.join(values)
            status_type, display_text, reference = classify_special_status(combined)
            if status_type is not None:
                special_statuses.setdefault(pos, {})[emp] = {
                    'type': status_type,
                    'display': display_text,
                    'reference': reference,
                }

    # Convert salary to numeric
    melted[salary] = pd.to_numeric(
        melted[salary].astype(str).str.extract(r'(\d+\.?\d*)', expand=False),
        errors='coerce'
    )
    melted = melted.dropna(subset=[salary])
    # Filter out empty titles
    melted = melted[melted[title].str.len() > 0]
    return melted, per_inspection, special_statuses


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


def generate_text_summary(df: pd.DataFrame) -> str:
    """Generate a text summary of compensation statistics for all positions."""
    from datetime import datetime
    lines = [
        "=" * 60,
        "COMPENSATION ANALYSIS SUMMARY REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    position_count = df[title].nunique()
    employer_count = df[location].nunique()
    lines.append(f"Positions analyzed:  {position_count}")
    lines.append(f"Employers compared:  {employer_count}")
    lines.append("-" * 60)
    lines.append("")

    for pos_name, group in df.groupby(title):
        all_salaries = pd.concat([group[sal_min], group[sal_max]])
        lines.extend([
            f"Position: {pos_name}",
            f"  Employers:   {len(group)}",
            f"  Min salary:  ${group[sal_min].min():,.2f}",
            f"  Max salary:  ${group[sal_max].max():,.2f}",
            f"  Median:      ${all_salaries.median():,.2f}",
            f"  Mean:        ${all_salaries.mean():,.2f}",
            "",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)


def graph(df, output, client_name: str = '', show_labels: bool = False, show_grid: bool = True):

    for name, group in df.groupby(title):
        # Sanitize name for filename by replacing slashes with underscores
        safe_name = name.replace('/', '_')

        group = group.sort_values(by=sal_max, ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        heights = group[sal_max] - group[sal_min]
        
        # Calculate y-axis range to determine appropriate thickness for zero-height bars
        all_salaries = pd.concat([group[sal_min], group[sal_max]])
        y_range = all_salaries.max() - all_salaries.min()
        # Use a fixed percentage of the y-axis range for consistent thickness
        ZERO_BAR_THICKNESS_RATIO = 0.02  # 2% of y-axis range
        zero_bar_height = y_range * ZERO_BAR_THICKNESS_RATIO if y_range > 0 else 1.0
        
        # Adjust heights for zero-height bars
        adjusted_heights = []
        for h in heights:
            if h == 0:
                adjusted_heights.append(zero_bar_height)
            else:
                adjusted_heights.append(h)
        
        linewidths = [3 if h == 0 else 1 for h in heights]
        bars = ax.bar(group[location], adjusted_heights, bottom=group[sal_min], color=group['color'], edgecolor='black', linewidth=linewidths, zorder=3)

        if show_labels:
            for bar, (_, row) in zip(bars, group.iterrows()):
                ax.annotate(
                    f'${row[sal_max]:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7,
                )

        ax.set_ylabel("Hourly Pay")
        ax.set_xlabel("Location")
        # ax.set_title(name)
        if show_grid:
            ax.grid(True, color="#AAA", zorder=0)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        plt.setp(ax.get_xticklabels(), ha="right")
        if client_name:
            fig.canvas.draw()
            for label in ax.get_xticklabels():
                if client_name in label.get_text():
                    label.set_color('#1565C0')
                    label.set_fontweight('bold')
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


def chart_to_svg(group: pd.DataFrame, name: str, client_name: str = '', show_labels: bool = False, show_grid: bool = True) -> str:
    """Generate a bar chart for a position group and return it as an inline SVG string."""
    group = group.sort_values(by=sal_max, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    heights = group[sal_max] - group[sal_min]
    
    # Calculate y-axis range to determine appropriate thickness for zero-height bars
    all_salaries = pd.concat([group[sal_min], group[sal_max]])
    y_range = all_salaries.max() - all_salaries.min()
    # Use a fixed percentage of the y-axis range for consistent thickness
    ZERO_BAR_THICKNESS_RATIO = 0.02  # 2% of y-axis range
    zero_bar_height = y_range * ZERO_BAR_THICKNESS_RATIO if y_range > 0 else 1.0
    
    # Adjust heights for zero-height bars
    adjusted_heights = []
    for h in heights:
        if h == 0:
            adjusted_heights.append(zero_bar_height)
        else:
            adjusted_heights.append(h)
    
    linewidths = [3 if h == 0 else 1 for h in heights]
    bars = ax.bar(group[location], adjusted_heights, bottom=group[sal_min], color=group['color'],
           edgecolor='black', linewidth=linewidths, zorder=3)

    if show_labels:
        for bar, (_, row) in zip(bars, group.iterrows()):
            ax.annotate(
                f'${row[sal_max]:,.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=7,
            )

    ax.set_ylabel("Hourly Pay")
    ax.set_xlabel("Location")
    #ax.set_title(name)
    if show_grid:
        ax.grid(True, color="#AAA", zorder=0)
    plt.xticks(rotation=60, ha='right', fontsize=8)
    if client_name:
        fig.canvas.draw()
        for label in ax.get_xticklabels():
            if client_name in label.get_text():
                label.set_color('#1565C0')
                label.set_fontweight('bold')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    svg_string = buf.read().decode('utf-8')
    # Strip XML declaration, keep just the <svg>...</svg> element
    svg_content = svg_string[svg_string.find('<svg'):]
    return svg_content


def graph_with_html(df, output_formats, client_name, input_file, per_inspection=None,
                    special_statuses=None,
                    show_labels: bool = False, show_grid: bool = True):
    """Generate graphs with optional HTML output"""
    # Generate image formats (non-HTML)
    image_formats = [fmt for fmt in output_formats if fmt != "html"]
    if image_formats:
        graph(df, image_formats, client_name=client_name, show_labels=show_labels, show_grid=show_grid)

    # Generate HTML if requested
    if 'html' in output_formats:
        generate_html_report(df, client_name, input_file,
                             per_inspection or {},
                             special_statuses or {},
                             show_labels=show_labels, show_grid=show_grid)


def generate_html_report(df, client_name, input_file, per_inspection=None,
                         special_statuses=None,
                         show_labels: bool = False, show_grid: bool = True):
    """Generate a self-contained HTML report with interactive Chart.js charts and filterable tables."""
    import json
    from datetime import datetime

    if per_inspection is None:
        per_inspection = {}
    if special_statuses is None:
        special_statuses = {}

    # Group data by position for HTML generation
    position_summaries = []

    for position_name, group in df.groupby(title):
        safe_name = position_name.replace('/', '_')

        # Sort by salary range
        sorted_group = group.sort_values(by=sal_max, ascending=True)

        # Create position summary (hourly employers) — these drive the chart
        chart_data = []  # only numeric/hourly rows go in the chart
        employers_data = []
        for _, row in sorted_group.iterrows():
            is_client = client_name in row[location]
            chart_data.append({
                'employer': row[location],
                'min': float(row[sal_min]),
                'max': float(row[sal_max]),
                'color': '#e8f4fd' if is_client else '#ffffff',
                'borderColor': '#1565C0' if is_client else '#333333',
                'is_client': is_client,
            })
            employers_data.append({
                'employer': row[location],
                'min_salary': row[sal_min],
                'max_salary': row[sal_max],
                'is_client': is_client,
                'per_inspection': False,
            })

        # Append per-inspection / fee-paid employers for this position (sorted by name)
        for pi_employer, pi_data in sorted(per_inspection.get(position_name, {}).items()):
            employers_data.append({
                'employer': pi_employer,
                'is_client': client_name in pi_employer,
                'per_inspection': True,
                'rates': pi_data['rates'],
                'fee_paid': pi_data.get('fee_paid', False),
            })

        # Append special-status employers (Cape Cod District, outsourced, see X)
        for ss_employer, ss_info in sorted(special_statuses.get(position_name, {}).items()):
            employers_data.append({
                'employer': ss_employer,
                'is_client': client_name in ss_employer,
                'per_inspection': False,
                'special_status': ss_info,
            })

        position_summaries.append({
            'name': position_name,
            'safe_name': safe_name,
            'employers': employers_data,
            'chart_data': chart_data,
        })

    # Serialise per-position chart data to JSON for embedding in HTML
    all_chart_data_json = json.dumps(
        {p['safe_name']: p['chart_data'] for p in position_summaries},
        ensure_ascii=False
    )

    # Generate HTML
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    input_stem = os.path.splitext(os.path.basename(input_file))[0]
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compensation Analysis Report - {client_name}</title>
    <!-- Chart.js for interactive bar charts -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
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
        .legend-toggle {{
            float: right;
            background: none;
            border: 1px solid #aaa;
            border-radius: 4px;
            width: 22px;
            height: 22px;
            line-height: 18px;
            text-align: center;
            font-size: 1.1em;
            color: #555;
            cursor: pointer;
            padding: 0;
            margin-top: 1px;
            transition: background 0.15s, color 0.15s;
        }}
        .legend-toggle:hover {{
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
            position: relative;
            margin: 20px 0;
            width: 100%;
            /* Use min-height instead of fixed height for better responsiveness */
            min-height: 400px; 
            height: auto;
            aspect-ratio: 16 / 9; /* Maintains a professional look */
            border: 1px solid #eee; /* Optional: adds definition */
            overflow: hidden; /* Prevents internal graph elements from leaking out */
        }}
        .chart-container img, .chart-container canvas {{
            width: 100%;
            height: 100%;
            object-fit: contain; /* Ensures the graph isn't distorted */
        }}
        /* ── Table ── */
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
        /* ── Sortable column headers ── */
        .sortable {{
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }}
        .sortable:hover {{
            background-color: #e9ecef;
        }}
        .sort-icon {{
            display: inline-block;
            margin-left: 4px;
            color: #aaa;
            font-size: 0.85em;
            vertical-align: middle;
        }}
        .sortable.sort-asc .sort-icon,
        .sortable.sort-desc .sort-icon {{
            color: #3498db;
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
        /* ── Special-status badges ── */
        .special-status-row td {{
            color: #555;
            font-style: italic;
        }}
        .status-badge {{
            display: inline-block;
            border-radius: 4px;
            padding: 1px 7px;
            font-size: 0.82em;
            font-style: normal;
            white-space: nowrap;
        }}
        .badge-district {{
            background: #cce5ff;
            border: 1px solid #004085;
            color: #004085;
        }}
        .badge-outsourced {{
            background: #fde8d8;
            border: 1px solid #c0392b;
            color: #c0392b;
        }}
        .badge-see {{
            background: #e8f5e9;
            border: 1px solid #2e7d32;
            color: #2e7d32;
        }}
        .badge-see a {{
            color: #1a5e20;
            text-decoration: underline;
        }}
        .badge-unknown {{
            background: #f5f5f5;
            border: 1px solid #888;
            color: #555;
        }}
        .fee-paid-badge {{
            display: inline-block;
            background: #e8f5e9;
            border: 1px solid #2e7d32;
            color: #2e7d32;
            border-radius: 4px;
            padding: 1px 7px;
            font-size: 0.82em;
            font-style: normal;
            white-space: nowrap;
        }}
        .badge-done-by {{
            background: #f3e5f5;
            border: 1px solid #7b1fa2;
            color: #7b1fa2;
        }}
        .badge-done-by a {{
            color: #4a0072;
            text-decoration: underline;
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
        /* ── Row hover highlight (from table or from chart bar) ── */
        .salary-table tbody tr {{
            transition: background-color 0.1s;
        }}
        .salary-table tbody tr.row-hover,
        .salary-table tbody tr.bar-hover {{
            background-color: #aeceaf !important;
            cursor: default;
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
        <h1>{input_stem}</h1>
        <h2>{client_name}, MA</h2>
        <h3>FY-26 Market Data</h3>
    </div>

    <!-- Legend / Table of Contents -->
    <div class="legend" id="legend">
        <h2>&#128269; Search<button class="legend-toggle" id="legendToggle" title="Show/hide">&#8722;</button></h2>
        <div id="legendBody">
        <input
            class="search-bar"
            id="positionSearch"
            type="search"
            placeholder="Search"
            autocomplete="off"
        />
        <ul class="legend-list" id="legendList">
'''

    for position in position_summaries:
        html_content += f'            <li class="legend-item" data-name="{position["name"].lower()}"><a href="#pos-{position["safe_name"]}">{position["name"]}</a></li>\n'

    html_content += '''        </ul>
        <p class="no-results" id="noResults" style="display:none;">No positions match your search.</p>
        </div>
    </div>

'''

    for position in position_summaries:
        html_content += f'''
    <div class="position-card" id="pos-{position['safe_name']}" data-position-name="{position['name'].lower()}">
        <div class="position-title">{position['name']}</div>
        
        <div class="chart-container">
            <canvas id="chart-{position['safe_name']}"></canvas>
        </div>

        <table class="salary-table" id="table-{position['safe_name']}">
            <thead>
                <tr>
                    <th class="sortable" data-col="employer" data-pos="{position['safe_name']}" onclick="sortTable('{position['safe_name']}', 'employer')">Employer <span class="sort-icon">&#8645;</span></th>
                    <th class="sortable" data-col="min"      data-pos="{position['safe_name']}" onclick="sortTable('{position['safe_name']}', 'min')">Minimum <span class="sort-icon">&#8645;</span></th>
                    <th class="sortable" data-col="max"      data-pos="{position['safe_name']}" onclick="sortTable('{position['safe_name']}', 'max')">Maximum <span class="sort-icon">&#8645;</span></th>
                </tr>
            </thead>
            <tbody>
'''

        for employer in position['employers']:
            if employer['per_inspection']:
                if employer.get('fee_paid'):
                    badge_html = '<span class="fee-paid-badge">&#128176; Fee Paid</span>'
                else:
                    rate_str = format_per_inspection_rate(employer['rates'])
                    badge_html = f'<span class="per-inspection-badge">&#128338; {rate_str}</span>'
                row_class = 'per-inspection-row' + (' client-row' if employer['is_client'] else '')
                html_content += f'''
                <tr class="{row_class}" data-employer="{employer['employer'].lower()}" data-min="" data-max="">
                    <td>{employer['employer']}</td>
                    <td colspan="2">{badge_html}</td>
                </tr>
'''
            elif 'special_status' in employer:
                ss = employer['special_status']
                badge = render_special_status_badge(ss, position_summaries)
                row_class = 'special-status-row' + (' client-row' if employer['is_client'] else '')
                html_content += f'''
                <tr class="{row_class}" data-employer="{employer['employer'].lower()}" data-min="" data-max="">
                    <td>{employer['employer']}</td>
                    <td colspan="2">{badge}</td>
                </tr>
'''
            else:
                row_class = 'client-row' if employer['is_client'] else ''
                html_content += f'''
                <tr class="{row_class}" data-employer="{employer['employer'].lower()}" data-min="{employer['min_salary']:.2f}" data-max="{employer['max_salary']:.2f}">
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

    # Embed chart data + all interactive JS
    html_content += f'''
    <button id="backToTop" title="Back to top">&#8679; Top</button>

    <script>
    // ── Embedded chart data (keyed by safe position name) ──────────────
    const ALL_CHART_DATA = {all_chart_data_json};
    const showGrid = {str(show_grid).lower()};

    // ── Chart registry ──────────────────────────────────────────────────
    const chartInstances = {{}};

    // Calculate appropriate thickness for zero-height bars based on y-axis range
    function calculateZeroBarThickness(rows) {{
        const allSalaries = [];
        rows.forEach(r => {{
            allSalaries.push(r.min);
            allSalaries.push(r.max);
        }});
        const yMin = Math.min(...allSalaries);
        const yMax = Math.max(...allSalaries);
        const yRange = yMax - yMin;
        const ZERO_BAR_THICKNESS_RATIO = 0.01; // 1% of y-axis range for consistent thickness
        return yRange * ZERO_BAR_THICKNESS_RATIO || 1.0;
    }}

    // ── Chart bar hover → highlight matching table row ───────────────────
    // barIndex: the Chart.js data index of the hovered bar, or -1 to clear.
    function highlightTableRow(safeName, barIndex) {{
        const table = document.getElementById('table-' + safeName);
        if (!table) return;
        const tbody = table.querySelector('tbody');
        if (!tbody) return;
        // Clear any existing bar-hover highlight
        tbody.querySelectorAll('tr.bar-hover').forEach(function(r) {{
            r.classList.remove('bar-hover');
        }});
        if (barIndex < 0) return;
        const chart = chartInstances[safeName];
        if (!chart) return;
        const employerName = (chart.data.labels[barIndex] || '').trim().toLowerCase();
        if (!employerName) return;
        // Find the matching table row by its employer cell text
        Array.from(tbody.querySelectorAll('tr')).forEach(function(row) {{
            const cell = row.querySelector('td');
            if (cell && cell.textContent.trim().toLowerCase() === employerName) {{
                row.classList.add('bar-hover');
            }}
        }});
    }}


    // ── Build Chart.js config object ────────────────────────────────────
    function buildChartConfig(rows, sortable, safeName) {{
        return {{
            type: 'bar',
            data: {{
                labels: rows.map(r => r.employer),
                datasets: [{{
                    label: 'Pay Range',
                    data: rows.map(r => r.min === r.max
                            ? [r.min - calculateZeroBarThickness(rows)/2, r.max + calculateZeroBarThickness(rows)/2]
                            : [r.min, r.max]),
                    backgroundColor: rows.map(r => r.color),
                    borderColor: rows.map(r => r.borderColor),
                    borderWidth: 1,
                    borderSkipped: false,
                }}]
            }},
            options: {{
                devicePixelRatio: 4,
                responsive: true,
                maintainAspectRatio: false,
                onHover: function(event, elements) {{
                    if (!safeName) return;
                    highlightTableRow(safeName, elements.length > 0 ? elements[0].index : -1);
                }},
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                // ctx.raw is the possibly-adjusted [lo, hi].
                                // Look up the original row to get the true values.
                                const dataIndex = ctx.dataIndex;
                                const orig = rows[dataIndex];
                                if (!orig) return '';
                                if (orig.min === orig.max)
                                    return ' $' + orig.min.toLocaleString('en-US', {{minimumFractionDigits:2}});
                                return ' $' + orig.min.toLocaleString('en-US', {{minimumFractionDigits:2}}) +
                                       ' \u2013 $' + orig.max.toLocaleString('en-US', {{minimumFractionDigits:2}});
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45,
                            font: function(context) {{
                                const isClient = rows[context.index] && rows[context.index].is_client;
                                return {{ size: 15, weight: isClient ? '800' : '400' }};
                            }},
                            color: '#222'
                        }},
                        grid: {{ display: showGrid, color: '#ddd' }}
                    }},
                    y: {{
                        beginAtZero: false,
                        grace: '8%',
                        title: {{ display: true, text: 'Hourly Pay', font: {{ size: 13 }} }},
                        ticks: {{ font: {{ size: 13 }} }},
                        grid: {{ display: showGrid, color: '#ddd' }}
                    }}
                }}
            }}
        }};
    }}

    function initChart(safeName) {{
        const canvas = document.getElementById('chart-' + safeName);
        if (!canvas) return;
        const rows = ALL_CHART_DATA[safeName] || [];
        const cfg = buildChartConfig(rows, true, safeName);
        chartInstances[safeName] = new Chart(canvas, cfg);
    }}

    // ── Reorder chart to match the current table row order ───────────────
    // orderedEmployers: array of employer names (original case) in table order
    function reorderChart(safeName, orderedEmployers) {{
        const chart = chartInstances[safeName];
        if (!chart) return;
        const allRows = ALL_CHART_DATA[safeName] || [];
        // Build lookup by trimmed lowercase name so leading/trailing whitespace
        // in employer names does not break the match against cell.textContent.
        const rowMap = {{}};
        allRows.forEach(r => {{ rowMap[r.employer.trim().toLowerCase()] = r; }});
        const ordered = orderedEmployers
            .map(e => rowMap[e.trim().toLowerCase()])
            .filter(Boolean);
        // Destroy and recreate so all scale options (beginAtZero, grace, etc.)
        // are guaranteed to be applied correctly on the new render.
        chart.destroy();
        const canvas = document.getElementById('chart-' + safeName);
        const cfg = buildChartConfig(ordered, true, safeName);
        chartInstances[safeName] = new Chart(canvas, cfg);
    }}

    // ── Sort state per position ──────────────────────────────────────────
    const sortState = {{}};   // {{ safeName: {{ col: 'employer'|'min'|'max', dir: 'asc'|'desc' }} }}

    function sortTable(safeName, col) {{
        const table = document.getElementById('table-' + safeName);
        if (!table) return;

        // Toggle direction or start fresh
        if (!sortState[safeName]) sortState[safeName] = {{ col: null, dir: 'asc' }};
        const state = sortState[safeName];
        if (state.col === col) {{
            state.dir = state.dir === 'asc' ? 'desc' : 'asc';
        }} else {{
            state.col = col;
            state.dir = 'asc';
        }}

        const tbody = table.querySelector('tbody');
        const rows  = Array.from(tbody.querySelectorAll('tr'));

        rows.sort(function(a, b) {{
            if (col === 'employer') {{
                const aV = (a.getAttribute('data-employer') || '').toLowerCase();
                const bV = (b.getAttribute('data-employer') || '').toLowerCase();
                const cmp = aV.localeCompare(bV);
                return state.dir === 'asc' ? cmp : -cmp;
            }} else {{
                const attr  = col === 'min' ? 'data-min' : 'data-max';
                const aStr  = a.getAttribute(attr);
                const bStr  = b.getAttribute(attr);
                // Rows without numeric values (badges) always sink to the bottom
                const aNum  = aStr !== '' && aStr !== null ? parseFloat(aStr) : null;
                const bNum  = bStr !== '' && bStr !== null ? parseFloat(bStr) : null;
                if (aNum === null && bNum === null) return 0;
                if (aNum === null) return 1;
                if (bNum === null) return -1;
                return state.dir === 'asc' ? aNum - bNum : bNum - aNum;
            }}
        }});

        // Re-append rows in sorted order
        rows.forEach(function(row) {{ tbody.appendChild(row); }});

        // Update sort indicator icons on headers
        table.querySelectorAll('.sortable').forEach(function(th) {{
            const icon = th.querySelector('.sort-icon');
            if (!icon) return;
            if (th.getAttribute('data-col') === col) {{
                icon.textContent = state.dir === 'asc' ? '\u2191' : '\u2193';
                th.classList.add(state.dir === 'asc' ? 'sort-asc' : 'sort-desc');
                th.classList.remove(state.dir === 'asc' ? 'sort-desc' : 'sort-asc');
            }} else {{
                icon.textContent = '\u21c5';
                th.classList.remove('sort-asc', 'sort-desc');
            }}
        }});

        // Collect numeric-row employers in their new visual order for chart
        const orderedEmployers = [];
        rows.forEach(function(row) {{
            const minVal = row.getAttribute('data-min');
            const maxVal = row.getAttribute('data-max');
            if (minVal !== '' && minVal !== null && maxVal !== '' && maxVal !== null) {{
                const cell = row.querySelector('td');
                if (cell) orderedEmployers.push(cell.textContent.trim());
            }}
        }});
        reorderChart(safeName, orderedEmployers);
    }}

    // ── Initialise all charts on load ────────────────────────────────────
    Object.keys(ALL_CHART_DATA).forEach(function(safeName) {{
        initChart(safeName);
    }});

    // ── Row hover → highlight matching chart bar ─────────────────────────
    // Highlight (or restore) the bar whose label matches employerName.
    function highlightChartBar(safeName, employerName, highlight) {{
        const chart = chartInstances[safeName];
        if (!chart) return;
        const labels = chart.data.labels;
        const idx = labels.findIndex(function(l) {{
            return l.trim().toLowerCase() === employerName.trim().toLowerCase();
        }});
        if (idx === -1) return;
        const ds = chart.data.datasets[0];
        if (highlight) {{
            ds.backgroundColor[idx] = '#aeceaf';  // sage green highlight
            ds.borderColor[idx]      = '#6a9e6b';  // darker sage border
        }} else {{
            // Restore original colours from ALL_CHART_DATA (unaffected by sort order)
            const allRows = ALL_CHART_DATA[safeName] || [];
            const origRow = allRows.find(function(r) {{
                return r.employer.trim().toLowerCase() === employerName.trim().toLowerCase();
            }});
            ds.backgroundColor[idx] = origRow ? origRow.color       : '#ffffff';
            ds.borderColor[idx]      = origRow ? origRow.borderColor : '#333333';
        }}
        chart.update('none');  // instant repaint, no animation
    }}

    // Attach mouseenter/mouseleave to every numeric tbody row.
    // Re-called after sort so newly inserted rows are also covered.
    function attachRowHoverHandlers() {{
        document.querySelectorAll('.salary-table tbody tr').forEach(function(row) {{
            // Skip rows that already have a listener attached
            if (row._hoverAttached) return;
            row._hoverAttached = true;
            // Only numeric rows have non-empty data-min/data-max
            const minVal = row.getAttribute('data-min');
            const maxVal = row.getAttribute('data-max');
            if (minVal === '' || minVal === null || maxVal === '' || maxVal === null) return;
            const table = row.closest('.salary-table');
            if (!table) return;
            const safeName = table.id.replace(/^table-/, '');
            const cell = row.querySelector('td');
            if (!cell) return;
            row.addEventListener('mouseenter', function() {{
                row.classList.add('row-hover');
                highlightChartBar(safeName, cell.textContent.trim(), true);
            }});
            row.addEventListener('mouseleave', function() {{
                row.classList.remove('row-hover');
                highlightChartBar(safeName, cell.textContent.trim(), false);
            }});
        }});
    }}

    // Initial attachment after charts are ready
    attachRowHoverHandlers();

    // ── Legend show/hide toggle ──────────────────────────────────────────────
    (function () {{
        const btn  = document.getElementById('legendToggle');
        const body = document.getElementById('legendBody');
        let open = true;
        btn.addEventListener('click', function () {{
            open = !open;
            body.style.display = open ? '' : 'none';
            btn.innerHTML = open ? '&#8722;' : '+';
            btn.title = open ? 'Hide' : 'Show';
        }});
    }})();

    // ── Position search (legend + cards) ────────────────────────────────
    (function () {{
        const searchInput  = document.getElementById('positionSearch');
        const legendItems  = document.querySelectorAll('.legend-item');
        const cards        = document.querySelectorAll('.position-card');
        const noResults    = document.getElementById('noResults');
        const backToTop    = document.getElementById('backToTop');

        searchInput.addEventListener('input', function () {{
            const query = this.value.trim().toLowerCase();
            let visibleCount = 0;

            legendItems.forEach(function (li) {{
                const name = li.getAttribute('data-name');
                if (!query || name.includes(query)) {{
                    li.classList.remove('legend-item-hidden');
                    visibleCount++;
                }} else {{
                    li.classList.add('legend-item-hidden');
                }}
            }});

            cards.forEach(function (card) {{
                const name = card.getAttribute('data-position-name');
                if (!query || name.includes(query)) {{
                    card.classList.remove('card-hidden');
                }} else {{
                    card.classList.add('card-hidden');
                }}
            }});

            noResults.style.display = visibleCount === 0 ? 'block' : 'none';
        }});

        window.addEventListener('scroll', function () {{
            backToTop.style.display = window.scrollY > 300 ? 'block' : 'none';
        }});

        backToTop.addEventListener('click', function () {{
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
            history.replaceState(null, '', window.location.pathname + window.location.search);
        }});
    }})();
    </script>
</body>
</html>
'''

    # Save HTML file
    os.makedirs('output/html', exist_ok=True)
    html_filename = f"{input_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
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


def process(
    file_path: str,
    output_formats: List[str],
    client_name: Optional[str] = None,
    show_labels: bool = False,
    show_grid: bool = True,
    generate_summary: bool = False,
) -> None:
    """
    Core processing pipeline: read data, build charts, save outputs.

    This function is the single entry-point for both ``main()`` (direct CLI
    invocation via ``main.py``) and ``cli.py``'s ``run_cli()`` so that neither
    needs to manipulate ``sys.argv``.

    Args:
        file_path: Path to the input compensation data file.
        output_formats: List of output format strings (e.g. ``['html', 'png']``).
        client_name: Name of the client/employer to highlight.  When ``None``
            the first employer column is used automatically.
        show_labels: Annotate each bar with its max-salary value.
        show_grid: Show grid lines on charts (default True).
        generate_summary: Print a statistics summary after generating charts.

    Raises:
        SystemExit: On unrecoverable errors (file not found, bad format, …).
    """
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Validate the input file                                              #
    # ------------------------------------------------------------------ #
    if not os.path.exists(file_path):
        print_error(
            f"File not found: {file_path}",
            "Check that the file path is correct. Example: -i input/csv/mydata.csv",
        )
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.csv', '.xls', '.xlsx', '.ods']:
        print_error(
            f"Unsupported file format: {ext}",
            "Supported formats: .csv, .xls, .xlsx, .ods",
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Detect column layout                                                 #
    # ------------------------------------------------------------------ #
    try:
        if ext == '.csv':
            first_line = pd.read_csv(file_path, nrows=1).columns.tolist()
            if all('Unnamed' in col or col.strip() == '' for col in first_line):
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
            "Make sure the file is not corrupted and has the correct format",
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Locate the position-title column                                     #
    # ------------------------------------------------------------------ #
    title_col: Optional[str] = None
    title_col_index: Optional[int] = None
    for i, col in enumerate(columns):
        if 'POSITION TITLE' in col.upper() or 'DARTMOUTH POSITION' in col.upper():
            title_col = col
            title_col_index = i
            break

    if title_col is None:
        print_error(
            "Could not find position title column in the file",
            "Make sure your data has a column with 'POSITION TITLE' or "
            "'DARTMOUTH POSITION TITLES' in its header",
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Auto-detect client name when not provided                           #
    # ------------------------------------------------------------------ #
    if client_name is None:
        if title_col_index is not None and title_col_index + 1 < len(columns):
            client_col = columns[title_col_index + 1]
            if 'Current' in client_col:
                client_name = client_col.split('Current')[0].strip()
            else:
                client_name = client_col
        else:
            print_error(
                "Client name not provided and could not be determined from file",
                "Use --client 'Client Name' to specify the client to highlight",
            )
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Load and transform data                                              #
    # ------------------------------------------------------------------ #
    try:
        df = read_data(file_path, ext, header_row)
    except Exception as e:
        print_error(
            f"Error reading data file: {e}",
            "Make sure the file format matches the extension",
        )
        sys.exit(1)

    # Normalise the title column name for downstream consistency
    if title_col != 'POSITION TITLE':
        df = df.rename(columns={title_col: 'POSITION TITLE'})

    df = remove_summary_columns(df)

    titles_present = df[title].notna() & (df[title].str.strip() != '')
    if titles_present.all():
        pass  # Every row has a title — no pairing needed
    else:
        if len(df) % 2 == 1:
            df = df.iloc[:-1]
        df = combine_lines(df)
        df = normalize(df)

    df, per_inspection, special_statuses = make_city_column(df)
    df = combine_high_low(df, client_name)
    graph_with_html(df, output_formats, client_name, file_path, per_inspection,
                    special_statuses=special_statuses,
                    show_labels=show_labels, show_grid=show_grid)

    logger.info("Charts generated successfully")
    print("\n" + "=" * 60)
    print("Success! Charts generated.")
    print("=" * 60)
    abs_output_path = os.path.abspath('output')
    abs_input_path = os.path.abspath(file_path)
    print(f"\nOutput location:\t{abs_output_path}/")
    print(f"Input file:\t\t{abs_input_path}")
    print(f"Client highlighted:\t{client_name}")
    print(f"Output formats:\t\t{', '.join(output_formats)}")
    print("=" * 60 + "\n")

    if generate_summary:
        summary = generate_text_summary(df)
        print(summary)


def main() -> None:
    """Parse CLI arguments and delegate to :func:`process`."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = FriendlyArgumentParser(description='Generate floating bar graphs for compensation data.')
    valid_formats = ['html', 'pdf', 'png', 'svg', 'jpg', 'jpeg', 'webp', 'eps']
    parser.add_argument('--client', type=str, help='Name of the client to be highlighted. Defaults to the first employer found in the data set', metavar='Employer')
    parser.add_argument('-i', type=str, default='input/csv/example_table.csv', help='Path to data file (supports .csv, .xls, .xlsx, .ods)', metavar='path/to/file')
    parser.add_argument('formats', nargs='*', help='Output format(s): html, pdf, png, svg, jpg, jpeg, webp, eps (default: png)', metavar='FORMAT')
    parser.add_argument('--show-labels', action='store_true', default=False, help='Annotate each bar with its max-salary value')
    parser.add_argument('--no-grid', action='store_true', default=False, help='Hide grid lines on charts')
    parser.add_argument('--summary', action='store_true', default=False, help='Print a statistics summary after generating charts')

    args = parser.parse_args()
    output_formats = args.formats if args.formats else ['png']

    # Validate formats
    invalid = [f for f in output_formats if f not in valid_formats]
    if invalid:
        parser.error(f"invalid format(s): {', '.join(invalid)}. Choose from: {', '.join(valid_formats)}")

    process(
        file_path=args.i,
        output_formats=output_formats,
        client_name=args.client,
        show_labels=args.show_labels,
        show_grid=not args.no_grid,
        generate_summary=args.summary,
    )


if __name__ == "__main__":
    main()
