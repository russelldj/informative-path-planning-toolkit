from pystac_client import Client
import planetary_computer as pc
import rioxarray
from pathlib import Path
import ubelt as ub
import os
import logging


def download_and_save(aoi, collection, output_folder: Path, asset="image"):
    # Search against the Planetary Computer STAC API
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    ub.ensuredir(output_folder)

    # Define your search with CQL2 syntax
    search = catalog.search(
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {"op": "s_intersects", "args": [{"property": "geometry"}, aoi]},
                {"op": "=", "args": [{"property": "collection"}, collection]},
            ],
        },
    )

    # Grab the first item from the search results and sign the assets
    items = search.get_all_items()
    for item in items:
        pc.sign_item(item).assets
        href = item.assets[asset].href
        name = Path(href).name
        savepath = Path(output_folder, asset + "_" + name)
        if os.path.isfile(savepath):
            logging.info(f"Skipping {savepath}")
            continue
        ds = rioxarray.open_rasterio(href)
        try:
            logging.info(f"Trying to save {href}")
            ds.rio.to_raster(savepath)
            logging.info(f"Saved {href}")
        except:
            continue

