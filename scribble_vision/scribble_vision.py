import sys
import constants
from gui import gui_utils, scribble_vision_window

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QTimer

import os
import logging

class ScribbleVision(QMainWindow, scribble_vision_window.Ui_MainWindow):
    """Main window of the NNSOA Emulator."""

    def __init__(self, parent=None):
        super(ScribbleVision, self).__init__(parent)
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='scribble_vision/scribble_vision.log',
                    filemode='w')

        # Test the logger
        logging.info('Program initialized.')

        # Initialize the filepath as the current directory
        self.filepath = os.getcwd()


        # Initialize a QTimer for continuous constant advancement
        self.total_frames = 0
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.advance)
        self.frame_timer.start(int(1000/constants.FRAMES_PER_SECOND))

        self.setupUi(self)
        self.initialize_gui()

    def test(self):
        logging.info("Test")
        self.statusbar.showMessage("Test")

    def run(self):
        logging.info("Run")
        self.statusbar.showMessage("Run")

    def advance(self):
        """ Advance by one frame. """
        self.total_frames += 1
        if self.total_frames % 100 == 0:
            self.statusbar.showMessage(f"Frames: {self.total_frames}")

    def setup_window(self):
        """ Setup the main window."""
        self.setWindowTitle(f"{constants.TITLE} {constants.VERSION}")
        self.setWindowIcon(QIcon(gui_utils.load_icon("application-icon.ico")))

    def on_exit(self):
        """ Close the application. """
        self.statusbar.showMessage("Exiting the NNSOA Emulator...")
        logging.info("Program Exited.")
        self.close()
    
    def initialize_gui(self):
        """ Initialize the GUI. """
        self.setup_window()
        self.statusbar.showMessage("Ready.")
        self.show()
       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScribbleVision()
    app.aboutToQuit.connect(window.on_exit)
    sys.exit(app.exec())


    