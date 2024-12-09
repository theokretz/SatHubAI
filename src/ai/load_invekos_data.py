from qgis.core import QgsVectorLayer, QgsProject, QgsRectangle, QgsVectorFileWriter, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsMessageLog, Qgis, QgsCoordinateTransformContext
import os

from ..utils import display_warning_message

class LoadInvekosData:
    def __init__(self):
        self._filepath = os.path.join(os.path.dirname(__file__), "INSPIRE_SCHLAEGE_2022_POLYGON.gpkg")

    def download_invekos_data(self, layer, directory):
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"
        filepath = os.path.join(directory, "output.gpkg")
        error = QgsVectorFileWriter.writeAsVectorFormatV3(layer, filepath, QgsCoordinateTransformContext(), options)

        if error[0] != QgsVectorFileWriter.NoError:
            QgsMessageLog.logMessage("Failed to download filtered layer.", level=Qgis.Critical)
            display_warning_message("Failed to download filtered layer.", "Downloading Failed!")
        else:
            QgsMessageLog.logMessage(f"Filtered Layer downloaded successfully at {filepath}.", level=Qgis.Info)

    def load_invekos_bounding_box(self, bbox, download_directory):
            # TODO: dynamically get invekos data
            layer = QgsVectorLayer(self._filepath, "Schläge 2022", "ogr")

            if not layer.isValid():
                QgsMessageLog.logMessage("Failed to load Invekos layer.", level=Qgis.Critical)
                display_warning_message("Failed to load Invekos layer.", "Loading Failed!")
                return

            if bbox == (None, None):
                QgsProject.instance().addMapLayer(layer)
            else:
                # transform coordinates
                source_crs = QgsCoordinateReferenceSystem("EPSG:4326")  # WGS84
                target_crs = layer.crs()  # EPSG:31287
                transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())

                bbox_min = transform.transform(bbox[0])
                bbox_max = transform.transform(bbox[1])
                bbox = QgsRectangle(bbox_min, bbox_max)

                # filter out everything outside bbox
                filtered_features = []
                for feature in layer.getFeatures():
                    if feature.geometry().intersects(bbox):
                        filtered_features.append(feature)

                QgsMessageLog.logMessage(f"{len(filtered_features)} plots found in your area.", level=Qgis.Info)

                # create a new temporary layer with filtered features
                if filtered_features:
                    crs_auth_id = target_crs.authid()
                    temp_layer = QgsVectorLayer(f"Polygon?crs={crs_auth_id}", "Filtered Layer Schläge 2022", "memory")
                    temp_provider = temp_layer.dataProvider()
                    temp_provider.addAttributes(layer.fields())
                    temp_layer.updateFields()
                    temp_provider.addFeatures(filtered_features)
                    if temp_layer.isValid():
                        QgsProject.instance().addMapLayer(temp_layer)
                        QgsMessageLog.logMessage("Filtered Layer loaded successfully.", level=Qgis.Info)

                        if download_directory:
                            self.download_invekos_data(temp_layer, download_directory)
                    else:
                        QgsMessageLog.logMessage("Failed to load filtered layer.", level=Qgis.Critical)
                        display_warning_message("Failed to load filtered layer.", "Loading Failed!")
                else:
                    QgsMessageLog.logMessage("No plots found in your selected area.")
                    display_warning_message("No plots found in your selected area.", "Change your area!")
