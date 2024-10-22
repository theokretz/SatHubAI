from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import QgsPointXY, QgsWkbTypes
from PyQt5.QtCore import Qt, pyqtSignal


class SelectArea(QgsMapToolEmitPoint):
    """lets users draw a rectangle on the QGIS map by clicking and dragging"""
    # define a signal that will emit the coordinates
    area_selected = pyqtSignal(QgsPointXY, QgsPointXY, QgsPointXY, QgsPointXY)

    def __init__(self, canvas):
        QgsMapToolEmitPoint.__init__(self, canvas)
        self.canvas = canvas
        self.rubber_band = None
        self.origin = None

    def canvasPressEvent(self, event):
        """mouse press event - triggers when the mouse button is pressed"""
        # set origin/first point and convert into coordinates
        self.origin = self.toMapCoordinates(event.pos())

        if not self.rubber_band:
            # create rubberband for drawing a rectangle
            self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
            self.rubber_band.setWidth(3)
            self.rubber_band.setColor(Qt.red)
            self.rubber_band.setFillColor(Qt.transparent)


    def canvasMoveEvent(self, event):
        """mouse move event - triggers when mouse is moved while mouse button is pressed"""
        if self.rubber_band and self.origin:
            # clear any previous rubberband points
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)

            # converts current mouse position into coordinates
            point = self.toMapCoordinates(event.pos())

            # drawing the rectangle/set points in rubberband
            self.rubber_band.addPoint(self.origin, False)   # update = False
            self.rubber_band.addPoint(QgsPointXY(point.x(), self.origin.y()), False)
            self.rubber_band.addPoint(point, False)
            self.rubber_band.addPoint(QgsPointXY(self.origin.x(), point.y()), True)  # update = True
            self.rubber_band.show()


    def canvasReleaseEvent(self, event):
        """mouse release event - triggers when mouse button is released"""
        if self.rubber_band and self.origin:
            # collect the coordinates from rubberband
            coords = [
                # 0 - The geometry index
                self.rubber_band.getPoint(0, 0),  # top-left
                self.rubber_band.getPoint(0, 1),  # top-right
                self.rubber_band.getPoint(0, 2),  # bottom-right
                self.rubber_band.getPoint(0, 3)   # bottom-left
            ]

            # emit coordinates
            self.area_selected.emit(*coords)

            # reset the rubberband and origin
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self.origin = None

