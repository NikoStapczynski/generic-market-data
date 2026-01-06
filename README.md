# Generic Town Market Data Graph Generator

This repository provides a generic template for generating floating bar graphs from town market data, adapted from a specific town's dataset.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your market data CSV file in the `data/csv/` directory. The file should have columns for position titles and salary ranges for different towns, similar to the sample data.

3. Update the data file to match your town's information. Ensure the column for your town is named `{town_name} Current {fy_year}` (e.g., "Melrose Current FY22").

4. If needed, adjust the `bad_columns` list in `floating_bar_graphs.py` to remove any summary columns specific to your data.

## Usage

Run the script with your town parameters:

```bash
python floating_bar_graphs.py --town_name "YourTown" --fy_year "FY23" --data_file "your_data.csv"
```

- `--town_name`: Name of your town (default: SampleTown)
- `--fy_year`: Fiscal year (default: FY22)
- `--data_file`: Name of your CSV file in data/csv/ (default: sample_market_data.csv)

The script will generate:
- Interactive HTML graphs in the `html/` directory
- PDF versions in the `pdf/` directory

## Data Format

The CSV should have:
- A 'POSITION TITLE' column
- Columns for each town with salary data (high/low ranges on alternating rows)
- Summary columns that will be removed (adjust in the script if needed)

## Adapting for a New Town

1. Replace the sample data with your town's data.
2. Rename columns to match `{town_name} Current {fy_year}`.
3. Update any hardcoded summary column names if they differ.
4. Run the script with your parameters.
