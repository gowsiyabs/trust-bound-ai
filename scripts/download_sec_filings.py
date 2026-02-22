"""Download SEC 10-K filings - modularized with CLI"""

import argparse
from pathlib import Path
from sec_edgar_downloader import Downloader

USER_EMAIL = "gowsiya.bs@gmail.com"

MAJOR_COMPANIES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V",
    "WMT", "PG", "MA", "HD", "DIS", "PYPL", "NFLX", "ADBE", "CRM", "INTC",
    "CSCO", "PFE", "TMO", "ABT", "NKE", "ORCL", "AVGO", "QCOM", "TXN", "AMD",
    "UNH", "CVX", "LLY", "MRK", "ABBV", "MDT", "BMY", "GILD", "AMGN", "ISRG",
    "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "USB", "PNC", "TFC",
    "XOM", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "HES", "OXY", "HAL",
    "T", "VZ", "TMUS", "CMCSA", "CHTR", "AMT", "CCI", "SBAC", "DISH", "DIS",
    "BA", "CAT", "GE", "HON", "MMM", "UPS", "FDX", "RTX", "LMT", "NOC",
    "KO", "PEP", "MDLZ", "MO", "PM", "CL", "KMB", "GIS", "K", "CPB",
    "COST", "TGT", "LOW", "SBUX", "MCD", "YUM", "CMG", "DPZ", "BKNG", "MAR",
]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download SEC 10-K filings")
    parser.add_argument("--max-companies", type=int, default=100,
                       help="Maximum number of companies (default: 100)")
    parser.add_argument("--filings-per-company", type=int, default=1,
                       help="Filings per company (default: 1)")
    parser.add_argument("--list", action="store_true",
                       help="List downloaded files")
    return parser.parse_args()


def list_downloaded_files(output_dir: Path):
    """List downloaded files"""
    files = list(output_dir.glob("**/*.txt"))
    print(f"\nDownloaded files: {len(files)}")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


def download_filings(max_companies: int, filings_per_company: int):
    """Download SEC filings"""
    output_dir = Path("data/documents")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dl = Downloader("SEC-EDGAR", USER_EMAIL, str(output_dir))
    
    companies = MAJOR_COMPANIES[:max_companies]
    
    print(f"Downloading {max_companies} companies, {filings_per_company} filing(s) each")
    
    for i, ticker in enumerate(companies, 1):
        try:
            print(f"[{i}/{len(companies)}] {ticker}...", end=" ")
            dl.get("10-K", ticker, limit=filings_per_company)
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
    
    print(f"\nComplete! Check: {output_dir}")


def main():
    """Main entry point"""
    args = parse_args()
    
    if args.list:
        list_downloaded_files(Path("data/documents"))
    else:
        download_filings(args.max_companies, args.filings_per_company)


if __name__ == "__main__":
    main()
