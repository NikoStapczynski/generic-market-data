# Compgrapher

Compgrapher (Compensation Grapher) is a tool for generating floating bar graphs from compensation market data for the purpose of comparison between employers.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/NikoStapczynski/compgrapher.git
   cd compgrapher
   ```

2. Set up Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Format your compensation data:

   The data file (CSV, XLS, XLSX, or ODS) should have a 'POSITION TITLE' column and columns for each employer with salary data. The data alternates between high and low salary ranges on consecutive rows for each position.

   Example structure (from `input/csv/example_table.csv`):

   | Row | POSITION TITLE | Employer A | Employer B | Employer C | Employer D | Employer E |
   |-----|----------------|------------|------------|------------|------------|------------|
   | 1   | Example Software Engineer | 100        | 95         | 105        |            |            |
   | 2   |                | 80         | 75         |            | 90         |            |
   | 3   | Example Data Scientist | 120        | 110        | 115        | 125        | 130        |
   | 4   |                | 90         | 85         | 95         |            | 120        |

   - Row 1 & 2: High and low salaries for Example Software Engineer
   - Row 3 & 4: High and low salaries for Example Data Scientist

   The script automatically removes summary columns (e.g., "Comp Data Points", "Comp Average"). Adjust the `bad_columns` list in the script if your data has different summary columns.

5. Generate sample PNG graphs:

   Run the script with the sample data to generate SVG graphs:
   ```bash
   python3 floating_bar_graphs.py --input input/csv/example_table.csv --output svg
   ```

   This will create SVG files in `output/svg/` showing floating bar graphs for each position.

   **Sample Output:**

   <img src="output/svg/Example%20Software%20Engineer.svg" width="50%" alt="Example Software Engineer">

   *Example Software Engineer: Floating bar graph comparing compensation between Employer A and Employer B*

   <br>

   <img src="output/svg/Example%20Data%20Scientist.svg" width="50%" alt="Example Data Scientist">

   *Example Data Scientist: Floating bar graph comparing compensation between Employer A and Employer B*

## Usage

Run the script with your client parameters:

```bash
python floating_bar_graphs.py --client "YourClient" --input "your_data.csv" --output html pdf png
```

- `--client Employer` (optional): Name of the employer to be highlighted. Defaults to the first employer found in the input file
- `--input path/to/file` (optional): Path to data file (supports .csv, .xls, .xlsx, .ods) (default: input/csv/example_table.csv)
- `--output file extension(s)` (optional): Output format(s) (default: html). Choices: html, pdf, png, svg, jpg, jpeg, webp, eps

The script will generate graphs in the specified formats in the `output/` subdirectories (e.g., `output/html/`, `output/pdf/`, etc.).

## Adapting for a New Client

1. Replace the sample data with your client's data.
2. Rename columns to match `{client} Current`.
3. Update any hardcoded summary column names if they differ.
4. Run the script with your parameters.
The script will generate graphs in the specified formats in the `output/` subdirectories (e.g., `output/html/`, `output/pdf/`, etc.).
