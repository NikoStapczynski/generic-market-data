# Generic Market Data Graph Generator

This repository provides a generic template for generating floating bar graphs from market data, adapted from a specific dataset.

## Setup

1. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your market data CSV file in the `input/csv/` directory. The file should have columns for position titles and salary ranges for different locations, similar to the sample data.

4. Update the data file to match your location's information. Ensure the column for your location is named `{location_name} Current {fy_year}` (e.g., "Melrose Current FY22").

5. If needed, adjust the `bad_columns` list in `floating_bar_graphs.py` to remove any summary columns specific to your data.

## Usage

Run the script with your location parameters:

```bash
python floating_bar_graphs.py --location_name "YourLocation" --fy_year "FY23" --data_file "your_data.csv" --output html pdf png
```

- `--location_name`: Name of your location (default: SampleLocation)
- `--fy_year`: Fiscal year (default: current fiscal year, e.g., FY26)
- `--data_file`: Name of your CSV file in input/csv/ (default: sample_market_data.csv)
- `--output`: Output formats (default: html). Choices: html, pdf, png, svg, jpg, jpeg, webp, eps

The script will generate graphs in the specified formats in the `output/` subdirectories (e.g., `output/html/`, `output/pdf/`, etc.).

## Data Format

The CSV should have a 'POSITION TITLE' column and columns for each location with salary data. The data alternates between high and low salary ranges on consecutive rows for each position.

Example structure:

| Row | POSITION TITLE | Location A | Location B |
|-----|----------------|------------|------------|
| 1   | Job Title 1    | 100        | 95         |
| 2   | Job Title 1    | 80         | 75         |
| 3   | Job Title 2    | 120        | 110        |
| 4   | Job Title 2    | 90         | 85         |

- Row 1 & 2: High and low salaries for Job Title 1
- Row 3 & 4: High and low salaries for Job Title 2

The script automatically removes summary columns (e.g., "Comp Data Points", "Comp Average"). Adjust the `bad_columns` list in the script if your data has different summary columns.

## Adapting for a New Location

1. Replace the sample data with your location's data.
2. Rename columns to match `{location_name} Current {fy_year}`.
3. Update any hardcoded summary column names if they differ.
4. Run the script with your parameters.
