import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import argparse

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

        group.sort_values(inplace=True, by=sal_max, ascending=True)
        fig = go.Figure([go.Bar(
            name=title,
            x=group[location],
            base=group[sal_min],
            y=group[sal_max]-group[sal_min],
            marker_color=group['color'],
            hovertemplate =
                '<b>%{x}</b>'+
                '<br>High: %{y:$,3f}'+
                '<br>Low: %{base:$,3f}'+
                '<extra></extra>',
            showlegend = False
        )])

        fig.update_traces(dict(marker_line_width=1, marker_line_color="black"))
        fig.update_yaxes(showgrid=True, gridcolor="#AAA", title_text="Pay Range (Hourly)")
        fig.update_xaxes(title_text="Location")
        fig.update_layout(title=name, template="simple_white")
        # fig.show()  # Commented out to prevent opening browser
        output_configs = {
            'html': ('output/html', lambda fig, path: fig.write_html(path)),
            'pdf': ('output/pdf', lambda fig, path: fig.write_image(path)),
            'png': ('output/png', lambda fig, path: fig.write_image(path)),
            'svg': ('output/svg', lambda fig, path: fig.write_image(path)),
            'jpg': ('output/jpg', lambda fig, path: fig.write_image(path)),
            'jpeg': ('output/jpeg', lambda fig, path: fig.write_image(path)),
            'webp': ('output/webp', lambda fig, path: fig.write_image(path)),
            'eps': ('output/eps', lambda fig, path: fig.write_image(path)),
        }
        for fmt in output:
            if fmt in output_configs:
                dir_path, write_func = output_configs[fmt]
                os.makedirs(dir_path, exist_ok=True)
                write_func(fig, f"{dir_path}/{safe_name}.{fmt}")
        


def main():
    global title
    parser = argparse.ArgumentParser(description='Generate floating bar graphs for compensation data.')
    parser.add_argument('--client', type=str, help='Name of the client to be highlighted. Defaults to the first employer found in the data set', metavar='Employer')

    parser.add_argument('--input', type=str, default='input/csv/sample_table.csv', help='Path to data file (supports .csv, .xls, .xlsx, .ods)', metavar='path/to/file')
    parser.add_argument('--output', nargs='+', default=['html'], choices=['html', 'pdf', 'png', 'svg', 'jpg', 'jpeg', 'webp', 'eps'], help='Output formats: html, pdf, png, svg, jpg, jpeg, webp, eps', metavar='file extension')

    args = parser.parse_args()

    file_path = args.input
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.csv', '.xls', '.xlsx', '.ods']:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Read columns to find title and possibly client
    if ext == '.csv':
        columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    elif ext in ['.xls', '.xlsx']:
        columns = pd.read_excel(file_path, nrows=0).columns.tolist()
    elif ext == '.ods':
        columns = pd.read_excel(file_path, engine='odf', nrows=0).columns.tolist()

    # Find the title column (contains 'POSITION TITLE')
    title = None
    title_col_index = None
    for i, col in enumerate(columns):
        if 'POSITION TITLE' in col.upper():
            title = col
            title_col_index = i
            break

    if title is None:
        raise ValueError("Could not find POSITION TITLE column in the file")

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
            raise ValueError("Client name not provided and could not determine from file")

    client_location = f'{args.client} Current'

    df = read_data(file_path, ext)
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
    graph(df, args.output)


if __name__ == "__main__":
    main()
