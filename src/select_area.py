from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import QgsPointXY, QgsWkbTypes
from PyQt5.QtCore import Qt, pyqtSignal
from .utils import display_warning_message

class SelectArea(QgsMapToolEmitPoint):
    """lets users draw a rectangle on the QGIS map by clicking and dragging"""
    # define a signal that will emit the coordinates
    area_selected = pyqtSignal(QgsPointXY, QgsPointXY, QgsPointXY, QgsPointXY)

    def __init__(self, canvas):
        QgsMapToolEmitPoint.__init__(self, canvas)
        self._canvas = canvas
        self._rubber_band = None
        self._origin = None

    def canvasPressEvent(self, event):
        """mouse press event - triggers when the mouse button is pressed"""
        # set origin/first point and convert into coordinates
        self._origin = self.toMapCoordinates(event.pos())

        if not self._rubber_band:
            # create rubberband for drawing a rectangle
            self._rubber_band = QgsRubberBand(self._canvas, QgsWkbTypes.PolygonGeometry)
            self._rubber_band.setWidth(3)
            self._rubber_band.setColor(Qt.red)
            self._rubber_band.setFillColor(Qt.transparent)


    def canvasMoveEvent(self, event):
        """mouse move event - triggers when mouse is moved while mouse button is pressed"""
        if self._rubber_band and self._origin:
            # clear any previous rubberband points
            self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)

            # converts current mouse position into coordinates
            point = self.toMapCoordinates(event.pos())

            # drawing the rectangle/set points in rubberband
            self._rubber_band.addPoint(self._origin, False)   # update = False
            self._rubber_band.addPoint(QgsPointXY(point.x(), self._origin.y()), False)
            self._rubber_band.addPoint(point, False)
            self._rubber_band.addPoint(QgsPointXY(self._origin.x(), point.y()), True)  # update = True
            self._rubber_band.show()


    def canvasReleaseEvent(self, event):
        """mouse release event - triggers when mouse button is released"""
        if self._rubber_band and self._origin:
            try:
                # collect the coordinates from rubberband
                coords = [
                    # 0 - The geometry index
                    self._rubber_band.getPoint(0, 0),  # top-left
                    self._rubber_band.getPoint(0, 1),  # top-right
                    self._rubber_band.getPoint(0, 2),  # bottom-right
                    self._rubber_band.getPoint(0, 3)   # bottom-left
                ]

                # check for invalid coordinates
                if any(coord is None for coord in coords):
                    raise ValueError("Invalid coordinates: some points are None")

                # emit coordinates
                self.area_selected.emit(*coords)
            except ValueError as e:
                display_warning_message("Please click and drag to select an area on the map.", "No area selected.")

            # reset the rubberband and origin
            self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self._origin = None

