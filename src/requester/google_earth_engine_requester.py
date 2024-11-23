
import ee
import requests
import os
import rasterio.features

def login():
    ee.Authenticate()
    ee.Initialize()
    print(ee.String('Hello!').getInfo())


def request_data(dataset, bands, start_date, end_date, region, output_path, scale=30, crs="EPSG:4326"):
    try:
        collection = (ee.ImageCollection(dataset) \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 30)) \
            .filterMetadata('DEGRADED_MSI_DATA_PERCENTAGE', 'equals', 0)

        least_cloudy_image = collection.first().select(['B4', 'B3', 'B2'])

        metadata = least_cloudy_image.getInfo()
        cloud_percentage = least_cloudy_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

        print(f"Cloudy Pixel: {cloud_percentage}%")


        collection_size = collection.size().getInfo()
        print(f"Number of images: {collection_size}")

        image = collection.median().select(bands)

        visualized_image = image.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)

        task = ee.batch.Export.image.toDrive(
            image= visualized_image,
            scale= 60,
            description='SingleGeoTIFF',
            folder='EarthEngineExports',
            fileNamePrefix='sentinel_image_new',
            region=region.getInfo()['coordinates'],
            fileFormat = 'GeoTIFF',
            crs=crs,
        )
        task.start()
        task.status()
        print('Downloaded image')
    except Exception as e:
        print(f"Error: {e}")


def start(self):
    login()
    dataset = "COPERNICUS/S2"
    bands = ["B4", "B3", "B2"]
    start_date = "2022-01-01"
    end_date = "2024-01-31"
    region = ee.Geometry.Polygon([
        [
            [16.1802, 48.1059],  # Bottom-left
            [16.5769, 48.1059],  # Bottom-right
            [16.5769, 48.3235],  # Top-right
            [16.1802, 48.3235],  # Top-left
            [16.1802, 48.1059]  # Close polygon
        ]
    ])
    output_path = (f"C:/Bachelorarbeit/myExportImageTask.tif")
    scale = 100  # 10-meter resolution
    crs = "EPSG:4326"  # WGS 84
    request_data(dataset, bands, start_date, end_date, region, output_path, scale, crs)