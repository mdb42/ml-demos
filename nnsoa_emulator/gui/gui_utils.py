
from importlib.resources import files
from PyQt6.QtGui import QImage, QPixmap, QCursor
from PyQt6 import QtCore

def load_icon(resource):
    resource_path = files("resources.icons") / resource
    with open(resource_path, 'rb') as file:
        icon_data = file.read()
    icon_image = QImage.fromData(icon_data)
    return QPixmap.fromImage(icon_image)

def load_cursor(resource, hot_x=0, hot_y=0):
    cursor_size = 18
    resource_path = files("resources.icons") / resource
    with open(resource_path, 'rb') as file:
        cursor_data = file.read()
    cursor_image = QImage.fromData(cursor_data).scaled(cursor_size, cursor_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    return QCursor(QPixmap.fromImage(cursor_image), hot_x, hot_y)
