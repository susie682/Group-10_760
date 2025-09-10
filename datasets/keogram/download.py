import requests
from bs4 import BeautifulSoup
import os

for year in range(2012, 2015):
    url = f"https://optics.gi.alaska.edu/pkr_asi_arch_aurorax_{year}.html"
    base = f"https://optics.gi.alaska.edu/pkr_asi_arch/aurorax_{year}/"

    save_dir = rf"C:.\{year}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Downloading {year} ===")
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        continue

    soup = BeautifulSoup(res.text, 'html.parser')
    img_tags = soup.find_all('img')

    for img in img_tags:
        img_url = img.get('src')
        full_url = img_url if img_url.startswith('http') else base + img_url

        try:
            r = requests.get(full_url, timeout=20)
            if r.status_code == 200:
                filename = os.path.join(save_dir, os.path.basename(img_url))
                with open(filename, 'wb') as f:
                    f.write(r.content)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed: {full_url} (status {r.status_code})")
        except Exception as e:
            print(f"Error downloading {full_url}: {e}")
