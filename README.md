# Compgrapher

Compgrapher is a tool for generating floating bar graphs from compensation market data.

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

3. Place your market data file anywhere in your filesystem. The file should have columns for position titles and salary ranges for different locations, similar to the sample data. Supported formats: .csv, .xls, .xlsx, .ods.

4. Update the data file to match your client's information. Ensure the column for your client is named `{client} Current {fy}` (e.g., "Melrose Current FY22").

5. If needed, adjust the `bad_columns` list in `floating_bar_graphs.py` to remove any summary columns specific to your data.

## Usage

Run the script with your client parameters:

```bash
python floating_bar_graphs.py --client "YourClient" --fy "FY23" --input "your_data.csv" --output html pdf png
```

- `--client Employer` (optional): Name of the employer to be highlighted. Defaults to the first employer found in the input file
- `--fy` (optional): Fiscal year (default: current fiscal year, e.g., FY26)
- `--input path/to/file` (optional): Path to data file (supports .csv, .xls, .xlsx, .ods) (default: input/csv/sample_market_data.csv)
- `--output file extension(s)` (optional): Output format(s) (default: html). Choices: html, pdf, png, svg, jpg, jpeg, webp, eps

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

## Adapting for a New Client

1. Replace the sample data with your client's data.
2. Rename columns to match `{client} Current {fy}`.
3. Update any hardcoded summary column names if they differ.
4. Run the script with your parameters.
