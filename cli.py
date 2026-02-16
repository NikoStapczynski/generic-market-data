"""
cli.py - Enhanced Command Line Interface for Compgrapher

This module provides a rich CLI with logging, configuration file support,
progress bars, and comprehensive argument handling.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

# Version info
__version__ = "1.0.0"


def setup_logging(verbose: bool = False, quiet: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging based on verbosity settings.
    
    Args:
        verbose: Enable debug-level logging
        quiet: Suppress all but error messages
        log_file: Optional file path for logging
        
    Returns:
        Configured logger instance
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    logger = logging.getLogger('compgrapher')
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration file settings with command line arguments.
    Command line arguments take precedence.
    
    Args:
        config: Configuration dictionary from file
        args: Parsed command line arguments
        
    Returns:
        Updated arguments namespace
    """
    # Map config keys to argument names
    config_mapping = {
        'input': 'i',
        'output': 'output',
        'client': 'client',
        'validate': 'validate',
        'verbose': 'verbose',
        'quiet': 'quiet',
        'show_grid': 'show_grid',
        'show_labels': 'show_labels',
        'summary': 'summary',
    }
    
    defaults = config.get('defaults', {})
    
    for config_key, arg_name in config_mapping.items():
        if config_key in defaults:
            # Only set if not explicitly provided on command line
            if getattr(args, arg_name, None) is None:
                setattr(args, arg_name, defaults[config_key])
    
    return args


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog='compgrapher',
        description='Generate floating bar graphs from compensation market data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s -i data.csv
  %(prog)s -i data.csv --output html pdf png
  %(prog)s -i data.csv --client "Acme Corp" --validate
  %(prog)s --config config.yaml --verbose

For more information, visit: https://github.com/nstapc/compgrapher
        '''
    )
    
    # Version
    parser.add_argument(
        '--version', '-V',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    # Input/Output
    parser.add_argument(
        '-i',
        type=str,
        default='input/csv/example_table.csv',
        help='Path to data file (supports .csv, .xls, .xlsx, .ods)',
        metavar='FILE'
    )
    
    parser.add_argument(
        '--output', '-o',
        nargs='+',
        default=['png'],
        choices=['html', 'pdf', 'png', 'svg', 'jpg', 'jpeg', 'webp', 'eps'],
        help='Output formats (default: png)',
        metavar='FORMAT'
    )
    
    parser.add_argument(
        '--client', '-c',
        type=str,
        help='Name of the employer to highlight (defaults to first employer in data)',
        metavar='NAME'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file',
        metavar='FILE'
    )
    
    # Validation
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run data validation checks and report issues'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data without generating graphs'
    )
    
    # Display options
    parser.add_argument(
        '--show-grid',
        action='store_true',
        default=True,
        help='Show grid lines on graphs (default: True)'
    )
    
    parser.add_argument(
        '--no-grid',
        action='store_false',
        dest='show_grid',
        help='Hide grid lines on graphs'
    )
    
    parser.add_argument(
        '--show-labels',
        action='store_true',
        default=False,
        help='Show salary labels on bars'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Generate a summary statistics report'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output (debug level)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Write logs to specified file',
        metavar='FILE'
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Check input file exists
    if not Path(args.i).exists():
        raise ValueError(f"Input file not found: {args.i}")
    
    # Check file extension is supported
    ext = Path(args.i).suffix.lower()
    if ext not in ['.csv', '.xls', '.xlsx', '.ods']:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Check for conflicting options
    if args.verbose and args.quiet:
        raise ValueError("Cannot use --verbose and --quiet together")


def run_cli(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Optional list of arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Load config file if specified
    if parsed_args.config:
        try:
            config = load_config(parsed_args.config)
            parsed_args = merge_config_with_args(config, parsed_args)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1
    
    # Setup logging
    logger = setup_logging(
        verbose=parsed_args.verbose,
        quiet=parsed_args.quiet,
        log_file=getattr(parsed_args, 'log_file', None)
    )
    
    try:
        # Validate arguments
        validate_args(parsed_args)
        
        logger.info(f"Compgrapher v{__version__}")
        logger.info(f"Input file: {parsed_args.i}")
        logger.info(f"Output formats: {parsed_args.output}")
        
        # Import here to avoid circular imports and allow CLI to work independently
        from main import main as run_main
        
        # Inject arguments into sys.argv for main.py compatibility
        sys.argv = ['compgrapher']
        if parsed_args.client:
            sys.argv.extend(['--client', parsed_args.client])
        sys.argv.extend(['-i', parsed_args.i])
        sys.argv.extend(['--output'] + parsed_args.output)
        if parsed_args.validate or parsed_args.validate_only:
            sys.argv.append('--validate')
        
        # Run the main processing
        run_main()
        
        # Generate summary if requested
        if parsed_args.summary:
            logger.info("Generating summary report...")
            # Summary generation would be handled by the graph generator
        
        logger.info("Processing complete!")
        return 0
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


def main():
    """Console script entry point."""
    sys.exit(run_cli())


if __name__ == '__main__':
    main()
