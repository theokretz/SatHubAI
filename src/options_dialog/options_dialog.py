# options_dialog.py

from PyQt5.QtWidgets import QDialog
from qgis.PyQt import uic
import os

# load ui file
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'options_dialog.ui'))

class OptionsDialog(QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(OptionsDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Satellite Provider Options")
