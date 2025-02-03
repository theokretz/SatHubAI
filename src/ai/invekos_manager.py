"""
invekos_manager.py
=================
Manages INVEKOS data loading and integration with satellite data processing.
"""

from typing import Optional, Tuple
from qgis.core import QgsVectorLayer, QgsProject
from .load_invekos_data import LoadInvekosData
import logging

logger = logging.getLogger("SatHubAI.InvekosManager")


class InvekosManager:
    """
    Manages INVEKOS data loading and coordinates with satellite data processing.
    """

    def __init__(self):
        self.loader = LoadInvekosData()
        self.current_layer = None
        self._bbox = None

    def load_invekos_data(self, coords: Tuple[float, float], start_date, end_date) -> Optional[QgsVectorLayer]:
        """
        Loads INVEKOS data for the specified area and time period.

        Parameters
        ----------
        coords : Tuple[float, float]
            Tuple of coordinates (top_left, bottom_right)
        start_date : QDate
            Start date for data
        end_date : QDate
            End date for data

        Returns
        -------
            QgsVectorLayer if successful, None otherwise
        """
        try:
            # Store bbox for later use
            self._bbox = coords

            # Load data using existing LoadInvekosData class
            self.loader.request_invekos_data(coords, start_date, end_date)

            # Get the layer that was just added
            self.current_layer = self.find_invekos_layer()

            return self.current_layer

        except Exception as e:
            logger.error(f"Failed to load INVEKOS data: {str(e)}")
            return None

    @staticmethod
    def find_invekos_layer() -> Optional[QgsVectorLayer]:
        """
        Finds the most recently added INVEKOS layer in the QGIS project.

        Returns
        -------
            QgsVectorLayer if successful, None otherwise
        """
        for layer in QgsProject.instance().mapLayers().values():
            if 'INVEKOS Layer' in layer.name():
                return layer
        return None

    def get_current_layer(self) -> Optional[QgsVectorLayer]:
        """
        Returns the current INVEKOS layer.

        Returns
        -------
            QgsVectorLayer if successful, None otherwise
        """
        if self.current_layer and self.current_layer.isValid():
            return self.current_layer

        # try to find layer if not set
        self.current_layer = self.find_invekos_layer()
        return self.current_layer
