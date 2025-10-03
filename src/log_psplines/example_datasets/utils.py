import os

import requests

from ..logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))


def download_file(url: str, local_path: str) -> None:
    """Download a file from a URL to a local path.

    Args:
        url: URL of the file to download
        local_path: Local path to save the downloaded file
    """
    response = requests.get(url)
    response.raise_for_status()  # Ensure it downloaded properly
    with open(local_path, "wb") as f:
        f.write(response.content)


PSD_URL = "https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt"
PSD_FILE = f"{HERE}/aligo_O3actual_H1.txt"
if not os.path.exists(PSD_FILE):
    logger.info(f"Downloading PSD file from {PSD_URL} to {PSD_FILE}")
    download_file(PSD_URL, PSD_FILE)
