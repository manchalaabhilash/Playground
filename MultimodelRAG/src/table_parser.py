"""
Table parsing implementation for advanced RAG systems.
Provides tools for extracting and processing tables from documents.
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import io
import numpy as np

class TableParser:
    """
    Extracts and processes tables from text documents.
    Handles various table formats including markdown, CSV, and ASCII tables.
    """
    
    def __init__(self):
        """Initialize the table parser"""
        # Regex patterns for different table formats
        self.markdown_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'
        self.ascii_pattern = r'(\+[-+]+\+\n(?:\|[^\n]+\|\n)+\+[-+]+\+)'
        self.csv_pattern = r'((?:[^,\n]+,){2,}[^,\n]+\n(?:(?:[^,\n]+,){2,}[^,\n]+\n)+)'
    
    def extract_tables(self, text: str) -> List[str]:
        """Extract tables from text"""
        tables = []
        
        # Extract markdown tables
        markdown_tables = re.findall(self.markdown_pattern, text)
        tables.extend(markdown_tables)
        
        # Extract ASCII tables
        ascii_tables = re.findall(self.ascii_pattern, text)
        tables.extend(ascii_tables)
        
        # Extract CSV-like tables
        csv_tables = re.findall(self.csv_pattern, text)
        tables.extend(csv_tables)
        
        return tables
    
    def parse_table(self, table_text: str) -> Optional[pd.DataFrame]:
        """Parse table text into a pandas DataFrame"""
        try:
            # Try to determine table format
            if '|' in table_text and '-' in table_text and '+' in table_text:
                # Likely ASCII table
                return self._parse_ascii_table(table_text)
            elif '|' in table_text and '-' in table_text:
                # Likely markdown table
                return self._parse_markdown_table(table_text)
            elif ',' in table_text:
                # Likely CSV table
                return self._parse_csv_table(table_text)
            else:
                # Unknown format
                return None
        except Exception as e:
            print(f"Error parsing table: {str(e)}")
            return None
    
    def _parse_markdown_table(self, table_text: str) -> pd.DataFrame:
        """Parse markdown table into DataFrame"""
        # Split into lines and remove empty lines
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        # Remove header separator line
        if len(lines) > 1 and all(c in '|:-' for c in lines[1]):
            lines = [lines[0]] + lines[2:]
        
        # Process each line
        rows = []
        for line in lines:
            # Remove leading/trailing |
            line = line.strip('|')
            # Split by | and strip whitespace
            cells = [cell.strip() for cell in line.split('|')]
            rows.append(cells)
        
        # Create DataFrame
        if rows:
            headers = rows[0]
            data = rows[1:]
            return pd.DataFrame(data, columns=headers)
        
        return pd.DataFrame()
    
    def _parse_ascii_table(self, table_text: str) -> pd.DataFrame:
        """Parse ASCII table into DataFrame"""
        # Split into lines and remove empty lines
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        # Remove border lines
        lines = [line for line in lines if not line.startswith('+')]
        
        # Process each line
        rows = []
        for line in lines:
            # Remove leading/trailing |
            line = line.strip('|')
            # Split by | and strip whitespace
            cells = [cell.strip() for cell in line.split('|')]
            rows.append(cells)
        
        # Create DataFrame
        if rows:
            headers = rows[0]
            data = rows[1:]
            return pd.DataFrame(data, columns=headers)
        
        return pd.DataFrame()
    
    def _parse_csv_table(self, table_text: str) -> pd.DataFrame:
        """Parse CSV table into DataFrame"""
        # Use pandas to parse CSV
        return pd.read_csv(io.StringIO(table_text))
    
    def table_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to markdown table text"""
        if df is None or df.empty:
            return ""
        
        # Convert DataFrame to markdown
        markdown = df.to_markdown(index=False)
        return markdown
    
    def search_table(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Search for information in a table based on a query"""
        if df is None or df.empty:
            return {"found": False, "result": None, "explanation": "Table is empty"}
        
        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()
        
        # Search in column names
        matching_columns = [col for col in df.columns if query_lower in str(col).lower()]
        
        # Search in cell values
        matching_rows = []
        for idx, row in df.iterrows():
            for col in df.columns:
                cell_value = str(row[col]).lower()
                if query_lower in cell_value:
                    matching_rows.append(idx)
                    break
        
        # Prepare results
        result = {
            "found": bool(matching_columns or matching_rows),
            "matching_columns": matching_columns,
            "matching_rows": df.iloc[matching_rows].to_dict('records') if matching_rows else [],
            "explanation": f"Found {len(matching_columns)} matching columns and {len(matching_rows)} matching rows"
        }
        
        return result