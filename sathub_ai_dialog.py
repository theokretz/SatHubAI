# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SatHubAIDialog
                                 A QGIS plugin
 This plugin automates the download of satellite data from multiple providers and detects specific features in the data with AI.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2024-10-16
        git sha              : $Format:%H$
        copyright            : (C) 2024 by Theo Kretz
        email                : theokretz2001@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os

# to load icons
from . import resources_rc

from PyQt5.QtWidgets import QMessageBox, QDockWidget, QWidget
from qgis.PyQt import uic
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsRasterLayer
)

from .select_area import SelectArea
from .sentinel_hub_request import true_color_without_clouds
from .utils import display_error_message

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'sathub_ai_dialog_base.ui'))

# QDockWidget needed for plugin to be docked
class SatHubAIDialog(QDockWidget, FORM_CLASS):
    def __init__(self, canvas, parent=None):
        """Constructor."""
        super(SatHubAIDialog, self).__init__(parent)

        # to select area on the map
        self.select_area_tool = None

        # to store coordinates
        self.coords_wgs84 = (None, None)

        # get canvas from parent - needed to interact with the map
        self.canvas = canvas

        # set up a widget to contain the UI elements
        self.main_widget = QWidget(self)
        self.setupUi(self.main_widget)

        # buttons
        self.pbSubmit.clicked.connect(self.on_pb_submit_clicked)
        self.tbSelectArea.clicked.connect(self.on_tb_select_area_clicked)

        self.setWindowTitle("SatHubAI")
        # ensure window is docked and not floating
        self.setFloating(False)

        # dock widget
        self.setWidget(self.main_widget)


    def on_pb_submit_clicked(self):
        """requests true color image"""
        start_date = self.calendarStart.selectedDate().toString("yyyy-MM-dd")
        end_date = self.calendarEnd.selectedDate().toString("yyyy-MM-dd")
        download_checked = self.cbDownload.isChecked()
        selected_file_type = self.comboboxFileType.currentText()

        if start_date > end_date:
            display_error_message('End date should be after start date.')
            return

        try:
            true_color_without_clouds(start_date, end_date, download_checked, selected_file_type, self.coords_wgs84)
        except Exception as e:
            display_error_message(str(e))


    def on_tb_select_area_clicked(self):
        """adds map layers and activates area drawing tool"""
        SatHubAIDialog.add_map_layer()

        # activate area drawing tool
        self.select_area_tool = SelectArea(self.canvas)
        self.canvas.setMapTool(self.select_area_tool)

        # connect the area_selected signal to the function that handles the coordinates
        self.select_area_tool.area_selected.connect(self.handle_coordinates)

    @staticmethod
    def add_map_layer():
        """adds map layers"""
        # TODO: choose map layer -> raster or vector
        project = QgsProject.instance()

        # remove existing layers
        if project.mapLayers() is not None:
            for layer in project.mapLayers():
                project.removeMapLayer(layer)

        # get current directory
        current_directory = os.path.dirname(__file__)

        # vector layer
        vector_layer_path = os.path.join(current_directory, 'mapLayers', 'world-administrative-boundaries', 'world-administrative-boundaries.shp')
        vector_layer = QgsVectorLayer(vector_layer_path, "Vector Layer", "ogr")

        if not vector_layer.isValid():
            raise FileNotFoundError(f"Could not load vector layer: {vector_layer_path}")
        else:
            print("Layer loaded successfully!")

        project.addMapLayer(vector_layer)
        '''
        # raster layer
        raster_layer_path = os.path.join(current_directory, 'mapLayers', 'NE1_LR_LC_SR_W', 'NE1_LR_LC_SR_W.tif')
        raster_layer = QgsRasterLayer(raster_layer_path, "Raster Layer")

        if not raster_layer.isValid():
            raise FileNotFoundError(f"Could not load raster layer: {raster_layer_path}")
        else:
            print("Layer loaded successfully!")

        project.addMapLayer(raster_layer)
        '''

    # TODO: maybe just send top left and bottom right coordinates
    def handle_coordinates(self, top_left, top_right, bottom_right, bottom_left):
        """saves coordinates and updates UI"""
        # save bounding box
        self.coords_wgs84 = (top_left, bottom_right)

        # updates UI with coordinates
        self.label_coordinates.setText(f"{top_left.x()}, {top_left.y()} \n{bottom_right.x()}, {bottom_right.y()}")
