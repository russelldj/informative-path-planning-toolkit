from ipp_toolkit.data.MaskedLabeledImage import (
    STACMaskedLabeledImage,
    torchgeoMaskedDataManger,
)

# item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2A_MSIL2A_20221125T112411_R037_T29TNF_20221126T161343"

# data_manager = STACMaskedLabeledImage(item_url, vis=VIS, downsample=8, blur_sigma=2)

torchgeo_data_manager = torchgeoMaskedDataManger(vis_all_chips=False)
torchgeo_data_manager.vis(vmin=0, vmax=10, cmap="tab10")

