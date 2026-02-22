"""PDF table extraction utilities"""

from typing import List, Dict, Any
import pdfplumber


def extract_tables_from_pdf(pdf_path: str) -> List[str]:
    """Extract tables from PDF"""
    tables_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:10]:  # First 10 pages
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        formatted = _format_table(table)
                        if formatted:
                            tables_text.append(formatted)
    except Exception:
        pass
    
    return tables_text


def _format_table(table: List[List[str]]) -> str:
    """Format table as text"""
    if not table or len(table) < 2:
        return ""
    
    rows = []
    for row in table:
        if row and any(cell for cell in row if cell):
            row_text = " | ".join(str(cell or "") for cell in row)
            rows.append(row_text)
    
    return "\n".join(rows) if rows else ""
