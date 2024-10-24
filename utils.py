# utils.py
from PyQt5.QtWidgets import QMessageBox

def display_error_message(error_message, error_title="An Error Occurred"):
    """displays error message"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(error_title)
    msg.setInformativeText(error_message)
    msg.setWindowTitle("Error")
    msg.exec_()


def display_warning_message(warning_message, warning_title="Action Needed"):
    """displays warning message"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(warning_title)
    msg.setInformativeText(warning_message)
    msg.setWindowTitle("Warning")
    msg.exec_()