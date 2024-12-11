# options_dialog.py
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog
from qgis.PyQt import uic
import os
from PyQt5.QtCore import Qt

from .options_config import OptionsConfig
from ..utils import display_error_message

# load ui file
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'options_dialog.ui'))

class OptionsDialog(QDialog, FORM_CLASS):
    icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../icons/login.svg"))
    collections_mapping = {
        "SENTINEL_HUB":["Sentinel-2 L1C", "Sentinel-2 L2A", "Landsat 1-5 MSS L1", "Landsat 4-5 TM L1", "Landsat 4-5 TM L2", "Landsat 7 ETM+ L1", "Landsat 7 ETM+ L2", "Landsat 8-9 OLI/TIRS L1", "Landsat 8-9 OLI/TIRS L2"],
        "PLANETARY_COMPUTER":["Sentinel-2 L2A", "Landsat Collection 2 L1", "Landsat Collection 2 L2"],
        "EARTH_SEARCH": [
            "Sentinel-2 L2A",
            (QIcon(icon_path), "Sentinel-2 L1C"),
            (QIcon(icon_path),"Landsat Collection 2 L2")
        ],
    }

    bands_mapping = {
        "Sentinel-2 L2A" : ["True Color", "False Color", "Red", "Green", "Blue", "Near Infrared"],
        "Sentinel-2 L1C" : ["True Color", "False Color", "Red", "Green", "Blue", "Near Infrared"],
        "Landsat Collection 2 L1" : ["False Color", "Red", "Green", "Near Infrared"],
        "Landsat Collection 2 L2" : ["True Color", "False Color", "Red", "Green", "Blue" , "Near Infrared"],
        "Landsat 1-5 MSS L1" : ["False Color", "Red", "Green", "Near Infrared"],
        "Landsat 4-5 TM L1" : ["True Color", "False Color", "Red", "Green", "Blue", "Near Infrared"],
        "Landsat 7 ETM+ L1": ["True Color", "False Color", "Red", "Green", "Blue", "Near Infrared"],
        "Landsat 7 ETM+ L2": ["True Color", "False Color", "Red", "Green", "Blue", "Near Infrared"],
        "Landsat 8-9 OLI/TIRS L1": ["True Color", "False Color", "Red", "Green", "Blue", "Near Infrared"],
        "Landsat 8-9 OLI/TIRS L2": ["True Color", "False Color", "Red", "Green", "Blue", "Near Infrared"],
    }

    options = pyqtSignal(OptionsConfig)

    def __init__(self,  parent=None, provider=None):
        """Constructor."""
        super(OptionsDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Satellite Provider Options")
        self._provider = provider

        # add collection items
        items = self.collections_mapping.get(self._provider, [])

        for item in items:
            if isinstance(item, tuple):
                self.comboboxCollection.addItem(item[0], item[1])
            else:
                self.comboboxCollection.addItem(item)

        self.comboboxCollection.currentIndexChanged.connect(self.on_collection_changed)
        self.pbSubmit.clicked.connect(self.on_pb_submit_clicked)

        # initialize the bands
        self.on_collection_changed()

    def on_collection_changed(self):
        """update the checkcbBands/Checkable Combobox of bands based on the selected collection"""
        selected_collection = self.comboboxCollection.currentText()
        bands = self.bands_mapping.get(selected_collection, [])

        self.checkcbBands.clear()

        for band in bands:
            item = self.checkcbBands.addItem(band)
            index = self.checkcbBands.findText(band)
            if band == "True Color" or (selected_collection == "Landsat Collection 2 L1" and band == "False Color"):
                self.checkcbBands.setItemCheckState(index, Qt.Checked)
            else:
                self.checkcbBands.setItemCheckState(index, Qt.Unchecked)

    def on_pb_submit_clicked(self):
        checked_bands = [self.checkcbBands.itemText(i) for i in range(self.checkcbBands.count())
                         if self.checkcbBands.itemCheckState(i) == Qt.Checked]
        if len(checked_bands) == 0:
            display_error_message("Please select at least one band", "No band selected!")
            return
        selected_options = OptionsConfig(self._provider, self.comboboxCollection.currentText(), self.cb_ndvi.isChecked(), checked_bands)
        self.options.emit(selected_options)
        self.close()


