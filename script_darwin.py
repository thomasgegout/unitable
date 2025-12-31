"""
Copyright (c) Docugami, Inc. All rights reserved.

Created by Louise Naud on 12/31/25
Description:
Usage      :
"""
import re
from pathlib import Path

from darwin.client import Client
from darwin.dataset.download_manager import _download_image
from dotenv import dotenv_values

secrets = dotenv_values(".env")
API_KEY = secrets["API_KEY"]


def safe_name(s: str) -> str:
    s = s.strip().replace("/", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def download_all_images(dataset_slug: str, out_dir: str = "darwin_images"):
    """
    dataset_slug: "team-slug/dataset-slug"
    Downloads ALL item source images (complete or not), no annotations.
    """
    client = Client.local()  # uses ~/.darwin/config.yaml
    ds = client.get_remote_dataset(dataset_slug)
    print(f"Downloading all images from {dataset_slug} {ds.dataset_id} to {out_dir}...")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # ds.workview_url_for_item()

    n = 0
    for item in ds.fetch_remote_files():  # <-- all items unless you pass filters  [oai_citation:3‡V7 Darwin Python SDK](https://darwin-py-sdk.v7labs.com/_modules/darwin/dataset/remote_dataset_v2.html)
        # Depending on dataset type, item.slots may be a list of Slot dataclasses or dict-like;
        # handle both defensively.
        print(f"Downloading {item.filename}...")
        url = ds.workview_url_for_item(item)
        print(f"URL: {url}")
        _download_image(url=url, path=out / item.filename, api_key=API_KEY)

        n += 1

    return n


if __name__ == "__main__":
    count = download_all_images("docugami/louise-green-tables", out_dir="all_images")
    print(f"Downloaded {count} files.")
 