import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf16/ssusi/data/edr-aurora/"
#https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf17/ssusi/data/edr-aurora/
#https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf18/ssusi/data/edr-aurora/
SAVE_BASE_DIR = Path(__file__).parent / "dmsp"

HEADERS = {"User-Agent": "Mozilla/5.0"}  # Simulate browser access


def get_links(url):
    """Get all hyperlinks from a webpage"""
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True)]


def download_file(url, save_path):
    """Download a single file"""
    if os.path.exists(save_path):
        print(f"Already exists, skipped: {save_path}")
        return
    r = requests.get(url, headers=HEADERS, stream=True)
    r.raise_for_status()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded: {save_path}")


def crawl_year(year):
    """Download all files of a given year"""
    year_url = f"{BASE_URL}{year}/"
    day_dirs = [d for d in get_links(year_url) if d.strip("/").isdigit()]  # 001/, 002/, ...
    
    for day_dir in day_dirs:
        day_url = f"{year_url}{day_dir}"
        file_links = [f for f in get_links(day_url) if f.endswith(".nc")]
        
        for file_link in file_links:
            file_url = f"{day_url}{file_link}"
            save_dir = os.path.join(SAVE_BASE_DIR, str(year), "16") #satellite number (16/17/18); change to match the number in line 5-7
            save_path = os.path.join(save_dir, file_link)
            download_file(file_url, save_path)


if __name__ == "__main__":
    for year in range(2014, 2017):  # Example: 2014–2016
        print(f"Starting download for year {year} ...")
        crawl_year(year)
    print("All downloads completed!")

