import logging
from qgis.core import Qgis, QgsMessageLog

class QgisLogHandler(logging.Handler):
    """Custom handler to send logs to the QGIS message log."""
    def emit(self, record):
        message = self.format(record)
        level = {
            "DEBUG": Qgis.Info,
            "INFO": Qgis.Info,
            "WARNING": Qgis.Warning,
            "ERROR": Qgis.Critical,
            "CRITICAL": Qgis.Critical
        }.get(record.levelname, Qgis.Info)
        QgsMessageLog.logMessage(message, 'SatHubAI', level)

def setup_logging():
    """Sets up logging for the SatHubAI plugin."""
    logger = logging.getLogger("SatHubAI")
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # QGIS Log Handler
        qgis_handler = QgisLogHandler()
        qgis_handler.setLevel(logging.DEBUG)
        qgis_formatter = logging.Formatter('%(name)s - %(message)s')
        qgis_handler.setFormatter(qgis_formatter)
        logger.addHandler(qgis_handler)

    return logger
