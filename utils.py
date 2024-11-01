# utils.py
from PyQt5.QtWidgets import QMessageBox
from qgis._core import QgsProject, QgsMessageLog, Qgis
from qgis.core import QgsRasterLayer

def display_error_message(error_message, error_title="An Error Occurred"):
    """displays error message"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(error_title)
    msg.setInformativeText(error_message)
    msg.setWindowTitle("Error")
    msg.exec_()


def display_warning_message(warning_message, warning_title="Action Needed"):
    """displays warning message"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(warning_title)
    msg.setInformativeText(warning_message)
    msg.setWindowTitle("Warning")
    msg.exec_()


def import_into_qgis(asset_url, layer_name):
    """loads the requested satellite image into QGIS"""
    raster_layer = QgsRasterLayer(asset_url, layer_name, "gdal")

    if not raster_layer.isValid():
        QgsMessageLog.logMessage("Layer failed to load!", level=Qgis.Critical)
        display_error_message("Image Layer failed to load!")
    else:
        QgsProject.instance().addMapLayer(raster_layer)
