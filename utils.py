# utils.py
from PyQt5.QtWidgets import QMessageBox

def display_error_message(error_message):
    """displays error message"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("An Error Occurred")
    msg.setInformativeText(error_message)
    msg.setWindowTitle("Error")
    msg.exec_()