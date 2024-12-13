"""
processor_factory
=================
Defines the ProcessorFactory class for creating processor instances based on satellite data collections.
"""
from .base_processor import Processor
from .landsat_processor import LandsatProcessor
from .sentinel_processor import SentinelProcessor
from ..request_config import RequestConfig
from qgis._core import QgsMessageLog, Qgis

class ProcessorFactory:
    """A factory class to create processor instances for handling satellite data collections."""

    _collection_processor_mapping = {
        "sentinel": SentinelProcessor,
        "landsat": LandsatProcessor,
    }

    @staticmethod
    def create_processor(config: RequestConfig, provider: str, collection: str) -> Processor:
        """
        Create a processor instance based on the provided collection.

        Parameters
        ----------
        config : RequestConfig
            Configuration object for the processor.
        provider : str
        collection : str

        Returns
        -------
        Processor
            An instance of the appropriate processor class.

        Raises
        ------
        ValueError
            If the collection is unsupported.
        """
        # go through the mapping and find the correct processor class
        for key, processor_class in ProcessorFactory._collection_processor_mapping.items():
            if collection.startswith(key):
                return processor_class(config, provider, collection)
        QgsMessageLog.logMessage(f"Unsupported collection: {collection}", level=Qgis.Critical)
        raise ValueError(f"Unsupported collection: {collection}")
