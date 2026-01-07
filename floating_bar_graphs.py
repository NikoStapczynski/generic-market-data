import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import argparse
import datetime

title = 'POSITION TITLE'
salary = 'salary'
location = 'location'
sal_min = 'salary_min'
sal_max = 'salary_max'

def read_data(file_path, ext):
    if ext == '.csv':
        return pd.read_csv(file_path,
                           dtype='float64',
                           converters={title: str})
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path, dtype='float64', converters={title: str})
    elif ext == '.ods':
        return pd.read_excel(file_path, engine='odf', dtype='float64', converters={title: str})
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def remove_summary_columns(df):
    bad_columns = [
        "Comp Data Points",
        "Comp Average",
        "Comp Lo-Hi Range",
        "Comp Median",
        "75th Percent of Market",
        "% Client Higher Lower than 75th Percentile"
    ]
    for c in bad_columns:
        df = df.drop(c, axis=1)

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
    return df.melt(
        id_vars=[title],
        value_vars=list(df.columns[1:]),
        value_name=salary,
        var_name=location
    ).dropna()


def combine_high_low(df, client_location):
    groups = df.groupby([location, title])

    def helper(group):
        minimum = group[salary].min()
        maximum = group[salary].max()

        if minimum == maximum:
            minimum = minimum * .99
            maximum = maximum

        client_color = '#AAA'
        default_color = '#FFF'

        if group[location].iloc[0] == client_location:
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
        fig.show()
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
                write_func(fig, f"{dir_path}/{name}.{fmt}")
        


def main():
    parser = argparse.ArgumentParser(description='Generate floating bar graphs for market data.')
    parser.add_argument('--location_name', type=str, default='SampleLocation', help='Name of the location for graphs')
    parser.add_argument('--fy_year', type=str, default=f'FY{datetime.datetime.now().year % 100:02d}', help='Fiscal year')
    parser.add_argument('--data_file', type=str, default='sample_market_data.csv', help='Data file name (supports .csv, .xls, .xlsx, .ods) in input/ subdirectories')
    parser.add_argument('--output', nargs='+', default=['html'], choices=['html', 'pdf', 'png', 'svg', 'jpg', 'jpeg', 'webp', 'eps'], help='Output formats: html, pdf, png, svg, jpg, jpeg, webp, eps')

    args = parser.parse_args()

    ext = os.path.splitext(args.data_file)[1].lower()
    if ext == '.csv':
        subdir = 'csv'
    elif ext in ['.xls', '.xlsx']:
        subdir = 'xls'
    elif ext == '.ods':
        subdir = 'ods'
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    file_path = f'input/{subdir}/{args.data_file}'
    client_location = f'{args.location_name} Current {args.fy_year}'

    df = read_data(file_path, ext)
    df = remove_summary_columns(df)
    df = combine_lines(df)
    df = normalize(df)
    df = make_city_column(df)
    df = combine_high_low(df, client_location)
    graph(df, args.output)


if __name__ == "__main__":
    main()
