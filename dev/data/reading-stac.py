import pystac_client
import planetary_computer
import pystac
import planetary_computer
import rioxarray
import matplotlib.pyplot as plt

item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2A_MSIL2A_20221125T112411_R037_T29TNF_20221126T161343"

# Load the individual item metadata and sign the assets
item = pystac.Item.from_file(item_url)
signed_item = planetary_computer.sign(item)

# Open one of the data assets (other asset keys to use: 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP', 'visual')
asset_href = signed_item.assets["visual"].href
ds = rioxarray.open_rasterio(asset_href)
print(dir(ds))
a = ds.to_masked_array()
ds
breakpoint()
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

time_range = "2020-12-01/2022-12-31"
bbox = [-122.2751, 47.5469, -121.9613, 47.7458]
area_of_interest = {
    "type": "Polygon",
    "coordinates": [
        [
            [41.217337, -8.527665],
            [41.216595, -8.527701],
            [41.215623, -8.527020],
            [41.216674, -8.526108],
            [41.217337, -8.527665],
        ]
    ],
}

options = dir(catalog)
options = [x for x in options if "collect" in x]
collections = list(catalog.get_collections())
naip = [x for x in collections if "planet" in x.id]
print(naip)

bbox = [-122.2751 + 1, 47.5469, -121.9613 + 1, 47.7458]
# bbox = [41.217337, -8.527665, 41.217337 + 2, -8.527665 + 2]
# search = catalog.search(collections=["landsat-c2-l2"], intersects=area_of_interest, datetime=time_range)
search = catalog.search(collections=["naip"], bbox=bbox, datetime=time_range)
items = search.get_all_items()
print(len(items))
