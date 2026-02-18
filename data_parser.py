"""
data_parser.py - Data parsing and validation module for Compgrapher
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass


class CompensationDataParser:
    """Parses and validates compensation data from various file formats"""
    
    SUPPORTED_FORMATS = {'.csv', '.xls', '.xlsx', '.ods'}
    BAD_COLUMNS = ['Comp Data Points', 'Comp Average', 'Average', 'Total']
    
    def __init__(self, filepath: str):
        """
        Initialize the parser with a file path.
        
        Args:
            filepath: Path to the compensation data file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if self.filepath.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {self.filepath.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file based on extension.
        
        Returns:
            DataFrame containing the raw compensation data
            
        Raises:
            Exception: If file cannot be read
        """
        logger.info(f"Loading data from {self.filepath}")
        
        try:
            suffix = self.filepath.suffix.lower()
            if suffix == '.csv':
                df = pd.read_csv(self.filepath)
            elif suffix in {'.xls', '.xlsx'}:
                df = pd.read_excel(self.filepath)
            elif suffix == '.ods':
                df = pd.read_excel(self.filepath, engine='odf')
            else:
                raise ValueError(f"Unexpected file format: {suffix}")
            
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove summary columns and clean the data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data")
        
        # Remove bad columns
        columns_to_drop = [col for col in df.columns if any(bad in col for bad in self.BAD_COLUMNS)]
        if columns_to_drop:
            logger.info(f"Removing columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # Find position title column - handle both 'POSITION TITLE' and 'DARTMOUTH POSITION TITLES'
        position_columns = ['POSITION TITLE', 'DARTMOUTH POSITION TITLES']
        found_column = None
        for col in position_columns:
            if col in df.columns:
                found_column = col
                logger.info(f"Using '{found_column}' as position title column")
                break
        
        if not found_column:
            raise DataValidationError(
                "Could not find position title column. "
                f"Expected one of: {', '.join(position_columns)}"
            )
        
        # Rename to standard 'POSITION TITLE' for consistency
        if found_column != 'POSITION TITLE':
            df = df.rename(columns={found_column: 'POSITION TITLE'})
        
        # Forward-fill position titles to handle alternating row format
        # where only the first row of each pair has a position title
        df['POSITION TITLE'] = df['POSITION TITLE'].ffill()
        
        # Remove rows where position title is still NaN (e.g., header rows with no data)
        df = df.dropna(subset=['POSITION TITLE'])
        
        logger.info(f"Data cleaned: {len(df)} rows remaining")
        return df
    
    def parse_compensation_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Parse compensation data into structured format.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary mapping position titles to employer compensation ranges
            Format: {position: {employer: (low, high)}}
        """
        logger.info("Parsing compensation data")
        
        compensation_data = {}
        employer_columns = [col for col in df.columns if col != 'POSITION TITLE']
        
        # Process pairs of rows (high/low)
        for i in range(0, len(df), 2):
            if i + 1 >= len(df):
                logger.warning(f"Odd number of rows, skipping last row")
                break
            
            position = df.iloc[i]['POSITION TITLE']
            if pd.isna(position):
                continue
            
            compensation_data[position] = {}
            
            for employer in employer_columns:
                high_val = df.iloc[i][employer]
                low_val = df.iloc[i + 1][employer]
                
                # Skip if either value is missing
                if pd.notna(high_val) and pd.notna(low_val):
                    try:
                        high = float(high_val)
                        low = float(low_val)
                        compensation_data[position][employer] = (low, high)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid numeric data for {position} - {employer}: "
                            f"high={high_val}, low={low_val}"
                        )
        
        logger.info(f"Parsed data for {len(compensation_data)} positions")
        return compensation_data
    
    def validate_data(self, compensation_data: Dict[str, Dict[str, Tuple[float, float]]]) -> List[str]:
        """
        Validate compensation data for logical consistency.
        
        Args:
            compensation_data: Parsed compensation data
            
        Returns:
            List of validation warnings
        """
        logger.info("Validating compensation data")
        warnings = []
        
        for position, employers in compensation_data.items():
            if not employers:
                warnings.append(f"Position '{position}' has no compensation data")
                continue
            
            for employer, (low, high) in employers.items():
                # Check if low > high
                if low > high:
                    warnings.append(
                        f"Position '{position}', Employer '{employer}': "
                        f"Low value ({low}) exceeds high value ({high})"
                    )
                
                # Check for negative values
                if low < 0 or high < 0:
                    warnings.append(
                        f"Position '{position}', Employer '{employer}': "
                        f"Negative compensation value detected"
                    )
                
                # Check for unusually large ranges (>100%)
                if high > 0 and (high - low) / high > 1.0:
                    warnings.append(
                        f"Position '{position}', Employer '{employer}': "
                        f"Unusually large range (>{100}%): ${low:,.0f} - ${high:,.0f}"
                    )
        
        if warnings:
            logger.warning(f"Found {len(warnings)} validation issues")
            for warning in warnings:
                logger.warning(warning)
        else:
            logger.info("Data validation passed")
        
        return warnings
    
    def process(self, validate: bool = True) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], List[str]]:
        """
        Full processing pipeline: load, clean, parse, and optionally validate.
        
        Args:
            validate: Whether to run validation checks
            
        Returns:
            Tuple of (compensation_data, validation_warnings)
        """
        df = self.load_data()
        df = self.clean_data(df)
        compensation_data = self.parse_compensation_data(df)
        
        warnings = []
        if validate:
            warnings = self.validate_data(compensation_data)
        
        return compensation_data, warnings


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = CompensationDataParser("input/csv/example_table.csv")
    data, warnings = parser.process()
    
    print(f"\nProcessed {len(data)} positions")
    if warnings:
        print(f"\nWarnings found: {len(warnings)}")
        for warning in warnings[:5]:  # Show first 5
            print(f"  - {warning}")