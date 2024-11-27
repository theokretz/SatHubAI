# options_dialog.py
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog
from qgis.PyQt import uic
import os

from .options_config import OptionsConfig

# load ui file
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'options_dialog.ui'))

class OptionsDialog(QDialog, FORM_CLASS):
    icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../icons/login.svg"))
    collections_list = {
        "SENTINEL_HUB":["Sentinel-2 L1C", "Sentinel-2 L2A", "Landsat 1-5 MSS L1", "Landsat 4-5 TM L1", "Landsat 4-5 TM L2", "Landsat 7 ETM+ L1", "Landsat 7 ETM+ L2", "Landsat 8-9 OLI/TIRS L1", "Landsat 8-9 OLI/TIRS L2"],
        "PLANETARY_COMPUTER":["Sentinel-2 L2A", "Landsat 2 L1", "Landsat 2 L2"],
        "EARTH_SEARCH": [
            "Sentinel-2 L2A",
            (QIcon(icon_path), "Sentinel-2 L1C"),
            "Landsat 2 L2"
        ],
    }
    options = pyqtSignal(OptionsConfig)

    def __init__(self,  parent=None, provider=None):
        """Constructor."""
        super(OptionsDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Satellite Provider Options")
        self._provider = provider

        # add collection items
        items = self.collections_list.get(self._provider, [])

        for item in items:
            if isinstance(item, tuple):
                self.comboboxCollection.addItem(item[0], item[1])
            else:
                self.comboboxCollection.addItem(item)


        self.pbSubmit.clicked.connect(self.on_pb_submit_clicked)


    def on_pb_submit_clicked(self):
        selected_options = OptionsConfig(self._provider, self.comboboxCollection.currentText(), self.cb_ndvi)
        self.options.emit(selected_options)
        self.close()


