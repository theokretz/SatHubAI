# invekos_retriever.py
from qgis.core import (
    QgsVectorLayer,
    QgsProject,
    QgsRectangle,
    QgsVectorFileWriter,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMessageLog,
    Qgis,
    QgsCoordinateTransformContext,
    QgsRendererCategory,
    QgsSymbol,
    QgsCategorizedSymbolRenderer
)
import random
from ..utils import calculate_bbox
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from PyQt5.QtGui import QColor
import logging
from datetime import datetime

logger = logging.getLogger("SatHubAI.InvekosRetriever")

class InvekosRetriever:
    def __init__(self):
        self.base_url = "https://gis.lfrz.gv.at/api/geodata/i009501/ogc/features/v1/collections/"
        self.full_url = None

    def request_invekos_data(self, coords, start_date, end_date):
        """
        Request INVEKOS data for a specified area and time period.

        Parameters
        ----------
        coords : list or tuple
            Bounding box coordinates [minx, miny, maxx, maxy]
        start_date : PyQt5.QtCore.QDate
            Start date for the data request
        end_date : PyQt5.QtCore.QDate
            End date for the data request

        Raises
        ------
        RequestException
            If the API request fails after all retries
        """

        end_dateset = self._get_invekos_dataset(end_date)
        self.full_url = self.base_url + end_dateset + "/items"
        start_dataset = self._get_invekos_dataset(start_date)

        if start_dataset != end_dateset:
            logger.warning(
                f"Warning: Your date range ({start_date.toString('yyyy-MM-dd')} to "
                f"{end_date.toString('yyyy-MM-dd')}) spans multiple datasets. "
                f"Using the most recent dataset from ({end_dateset})."
            )

        bbox = calculate_bbox(coords)
        max_workers = os.cpu_count() * 2

        # fetch all features within the bounding box with pagination
        all_features = self._fetch_all_with_pagination(bbox, limit=100, max_workers=max_workers)

        if not all_features:
            logger.error("No data available for requested time range")
            raise

        logger.info(f"Total Invekos features fetched: {len(all_features)}")

        # save as GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "features": all_features
        }

        file_name = f'invekos_{end_dateset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.geojson'
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)

        self._add_layer_to_qgis(file_name, f'INVEKOS Layer {end_dateset}')

    @staticmethod
    def _get_invekos_dataset(date):
        """
        Determine the appropriate INVEKOS dataset identifier for a given date.

        Parameters
        ----------
        date : PyQt5.QtCore.QDate
            The date for which to determine the appropriate dataset

        Returns
        -------
        str
            Dataset identifier in the format 'i009501:invekos_schlaege_YYYY_polygon' or
            'i009501:invekos_schlaege_YYYY_V_polygon' where YYYY is the year and V is
            the version (1 or 2)

        Raises
        ------
        ValueError
            If the date is before July 2015 when data coverage begins
        """
        month = date.month()
        year = date.year()

        # there is no INVEKOS Schläge data available before 30.6.2015
        if year < 2015:
            logger.warning(f"No data available for year {year}. Available data starting from July 2015, this data is being used.")
            return f"i009501:invekos_schlaege_2015_polygon"
        elif year == 2015 and month <= 6:
            logger.warning(f"No data available for year {year}. Available data starting from July 2015,, this data is being used.")
            return f"i009501:invekos_schlaege_2015_polygon"

        # before 2022 a year ranged from 30.6.2015-30.6.2016
        if year < 2022:
            if month <= 6:
                return f"i009501:invekos_schlaege_{year - 1}_polygon"
            return f"i009501:invekos_schlaege_{year}_polygon"

        # special case for 2022
        if year == 2022:
            # first half of 2022 gets covered by 2021's dataset
            if month <= 6:
                return "i009501:invekos_schlaege_2021_polygon"
            return "i009501:invekos_schlaege_2022_polygon"

        # from 2023 onwards, the year is split into two datasets: 2023-1 and 2023-2
        # 2023-1: 30.6.2023-31.12.2023 and 2023-2: 31.12.2023-30.6.2024
        if month <= 6:
            return f"i009501:invekos_schlaege_{year - 1}_2_polygon"
        return f"i009501:invekos_schlaege_{year}_1_polygon"


    def _add_layer_to_qgis(self, filepath, layer_name):
        """
        Adds a GeoJSON file as a layer in QGIS and assigns each unique crop type a color.

        Parameters
        ----------
        filepath : str
            Path to the GeoJSON file.
        layer_name : str
            Name of the layer.
        """
        layer = QgsVectorLayer(filepath, layer_name, "ogr")

        if not layer.isValid():
            print(f"Failed to load the layer from {filepath}")
            return

        # change tooltip to "snar_bezeichnung" - crop type
        expression = "snar_bezeichnung"
        layer.setDisplayExpression(expression)

        # apply unique colors to each crop type
        self._apply_unique_colors(layer)

        QgsProject.instance().addMapLayer(layer)
        logger.info(f"Invekos layer added to QGIS: {layer_name}")

    @staticmethod
    def _apply_unique_colors(layer):
        """
        Apply unique random colors to different crop types (snar_bezeichnung).

        Parameters
        ----------
        layer : QgsVectorLayer
            The QGIS vector layer to be styled
       """
        unique_values = layer.uniqueValues(layer.fields().indexFromName("snar_bezeichnung"))
        categories = []
        for crop_type in unique_values:
            # create random color
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

            # create category
            category = QgsRendererCategory(crop_type, symbol, str(crop_type))
            categories.append(category)

        renderer = QgsCategorizedSymbolRenderer("snar_bezeichnung", categories)
        layer.setRenderer(renderer)

    def _fetch_items(self, bbox, start_index, limit, retries=3, delay=2):
        """
        Fetch items within the bounding box (bbox) and pagination startIndex.

        Parameters
        ----------
        bbox : list or tuple
            Bounding box coordinates [minx, miny, maxx, maxy]
        start_index : int
            Pagination starting Index.
        limit : int
            Number of items per request.
            The Maximum for this API is 100.
        retries : int
            Number of retry attempts (default: 3).
        delay : int
            Delay in seconds between retries (default: 2).

        Returns
        -------
        list
            List of GeoJSON features retrieved from the API.
        """
        params = {
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "startIndex": start_index,
            "limit": limit
        }

        for attempt in range(retries):
            try:
                response = requests.get(self.full_url, params=params, timeout=10)
                response.raise_for_status()  # this will raise an exception for bad status codes
                data = response.json()
                features = data.get("features", [])
                return features
            except requests.exceptions.RequestException as e:
                logger.error(f"Error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

        logger.error(f"Failed to fetch data after {retries} retries (startIndex={start_index}).")
        return []

    def _fetch_all_with_pagination(self, bbox, limit=100, max_workers=5):
        """
        Fetch all items within the bounding box (bbox) using concurrency and pagination.

        Parameters
        ----------
        bbox : Iterable
            [minx, miny, maxx, maxy] coordinates.
        limit : int
            Number of items per request.
        max_workers : int
            Number of concurrent threads.
        """
        all_features = []
        futures = []
        start_index = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                future = executor.submit(self._fetch_items, bbox, start_index, limit)
                futures.append(future)
                start_index += limit

                # stop dynamically when a batch returns fewer than `limit` items
                if len(futures) > 0:
                    last_result = futures[-1].result()
                    if len(last_result) < limit:
                        break

            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_features.extend(result)

        return all_features
