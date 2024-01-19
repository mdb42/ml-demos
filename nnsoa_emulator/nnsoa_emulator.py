import sys
import constants
from gui import gui_utils, nnsoa_emulator_window

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QIcon, QIntValidator
from PyQt6.QtCore import QTimer

import os
import logging

class NNSOAEmulator(QMainWindow, nnsoa_emulator_window.Ui_MainWindow):
    """Main window of the NNSOA Emulator."""

    def __init__(self, parent=None):
        super(NNSOAEmulator, self).__init__(parent)
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='nnsoa_emulator/nnsoa_emulator.log',
                    filemode='w')

        # Test the logger
        logging.info('Program initialized.')

        # Initialize the filepath as the current directory
        self.filepath = os.getcwd()

        # Hyperparameters
        self.inputs_count = constants.DEFAULT_INPUT_COUNT
        self.outputs_count = constants.DEFAULT_OUTPUT_COUNT
        self.hidden_nodes_count = constants.DEFAULT_HIDDEN_NODES_COUNT
        self.hidden_gen_count = constants.DEFAULT_HIDDEN_GEN_COUNT
        self.generations_count = constants.DEFAULT_GENERATIONS_COUNT
        self.training_obs_count = constants.DEFAULT_TRAINING_OBS_COUNT
        self.population_count = constants.DEFAULT_POPULATION_COUNT
        self.jump_count = constants.DEFAULT_JUMP_COUNT

        # Flags
        self.reduce_check = constants.DEFAULT_REDUCE_CHECK
        self.classification_check = constants.DEFAULT_CLASSIFICATION_CHECK
        self.logistic_function_check = constants.DEFAULT_LOGISTIC_FUNCTION_CHECK
        self.more_check = constants.DEFAULT_MORE_CHECK
        self.converge_check = constants.DEFAULT_CONVERGE_CHECK

        # Files
        self.input_file = constants.DEFAULT_INPUT_FILE
        self.error_file = constants.DEFAULT_ERROR_FILE
        self.config_file = constants.DEFAULT_CONFIG_FILE
        self.weights_file = constants.DEFAULT_WEIGHTS_FILE
        
        # Testing
        self.testing_obs_count = constants.DEFAULT_TESTING_OBS_COUNT
        self.test_file = constants.DEFAULT_TEST_FILE

        # Output
        self.output_file = constants.DEFAULT_OUTPUT_FILE

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
        self.output_browser.append("Testing parameters...")
        self.output_browser.append(self.generate_parameter_report())
        self.output_browser.append("Testing parameters finished.")
        self.output_browser.verticalScrollBar().setValue(self.output_browser.verticalScrollBar().maximum())

    def run(self):
        logging.info("Run")
        self.statusbar.showMessage("Run")
        self.output_browser.append("Running parameters...")
        self.output_browser.append(self.generate_parameter_report())
        self.output_browser.append("Running parameters finished.")
        self.output_browser.verticalScrollBar().setValue(self.output_browser.verticalScrollBar().maximum())

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
    
    def update_inputs_count(self):
        try:
            self.inputs_count = int(self.inputs_line_edit.text())
        except ValueError:
            self.inputs_count = constants.DEFAULT_INPUT_COUNT
            self.inputs_line_edit.setText(str(constants.DEFAULT_INPUT_COUNT))
            self.statusbar.showMessage("Invalid input for inputs count. Reverting to default.")
            
    def update_outputs_count(self):
        try:
            self.outputs_count = int(self.outputs_line_edit.text())
        except ValueError:
            self.outputs_count = constants.DEFAULT_OUTPUT_COUNT
            self.outputs_line_edit.setText(str(constants.DEFAULT_OUTPUT_COUNT))
            self.statusbar.showMessage("Invalid input for outputs count. Reverting to default.")

    def update_hidden_nodes_count(self):
        try:
            self.hidden_nodes_count = int(self.hidden_nodes_line_edit.text())
        except ValueError:
            self.hidden_nodes_count = constants.DEFAULT_HIDDEN_NODES_COUNT
            self.hidden_nodes_line_edit.setText(str(constants.DEFAULT_HIDDEN_NODES_COUNT))
            self.statusbar.showMessage("Invalid input for hidden nodes count. Reverting to default.")

    def update_hidden_gen_count(self):
        try:
            self.hidden_gen_count = int(self.hidden_gen_line_edit.text())
        except ValueError:
            self.hidden_gen_count = constants.DEFAULT_HIDDEN_GEN_COUNT
            self.hidden_gen_line_edit.setText(str(constants.DEFAULT_HIDDEN_GEN_COUNT))
            self.statusbar.showMessage("Invalid input for hidden gen count. Reverting to default.")

    def update_generations_count(self):
        try:
            self.generations_count = int(self.generations_line_edit.text())
        except ValueError:
            self.generations_count = constants.DEFAULT_GENERATIONS_COUNT
            self.generations_line_edit.setText(str(constants.DEFAULT_GENERATIONS_COUNT))
            self.statusbar.showMessage("Invalid input for generations count. Reverting to default.")

    def update_training_obs_count(self):
        try:
            self.training_obs_count = int(self.training_obs_line_edit.text())
        except ValueError:
            self.training_obs_count = constants.DEFAULT_TRAINING_OBS_COUNT
            self.training_obs_line_edit.setText(str(constants.DEFAULT_TRAINING_OBS_COUNT))
            self.statusbar.showMessage("Invalid input for training obs count. Reverting to default.")

    def update_reduce_check(self):
        self.reduce_check = self.reduce_checkbox.isChecked()

    def update_classification_check(self):
        self.classification_check = self.classification_checkbox.isChecked()

    def update_logistic_function_check(self):
        self.logistic_function_check = self.logistic_function_checkbox.isChecked()

    def update_more_check(self):
        self.more_check = self.more_checkbox.isChecked()

    def update_converge_check(self):
        self.converge_check = self.converge_checkbox.isChecked()

    def update_population_count(self):
        try:
            self.population_count = int(self.population_line_edit.text())
        except ValueError:
            self.population_count = constants.DEFAULT_POPULATION_COUNT
            self.population_line_edit.setText(str(constants.DEFAULT_POPULATION_COUNT))
            self.statusbar.showMessage("Invalid input for population count. Reverting to default.")

    def update_jump_count(self):
        try:
            self.jump_count = int(self.jump_line_edit.text())
        except ValueError:
            self.jump_count = constants.DEFAULT_JUMP_COUNT
            self.jump_line_edit.setText(str(constants.DEFAULT_JUMP_COUNT))
            self.statusbar.showMessage("Invalid input for jump count. Reverting to default.")

    def update_input_file(self):
        self.input_file = self.input_file_line_edit.text()

    def update_error_file(self):
        self.error_file = self.error_file_line_edit.text()

    def update_config_file(self):
        self.config_file = self.config_file_line_edit.text()

    def update_weights_file(self):
        self.weights_file = self.weights_file_line_edit.text()

    def update_testing_obs_count(self):
        self.testing_obs_count = self.testing_obs_line_edit.text()

    def update_test_file(self):
        self.test_file = self.test_file_line_edit.text()

    def update_output_file(self):
        self.output_file = self.output_file_line_edit.text()
    
    def initialize_gui(self):
        """ Initialize the GUI. """
        self.setup_window()
        self.test_button.clicked.connect(self.test)
        self.run_button.clicked.connect(self.run)

        # Set the default values
        self.inputs_line_edit.setText(str(self.inputs_count))
        self.outputs_line_edit.setText(str(self.outputs_count))
        self.hidden_nodes_line_edit.setText(str(self.hidden_nodes_count))
        self.hidden_gen_line_edit.setText(str(self.hidden_gen_count))
        self.generations_line_edit.setText(str(self.generations_count))
        self.training_obs_line_edit.setText(str(self.training_obs_count))
        self.reduce_checkbox.setChecked(self.reduce_check)
        self.classification_checkbox.setChecked(self.classification_check)
        self.logistic_function_checkbox.setChecked(self.logistic_function_check)
        self.more_checkbox.setChecked(self.more_check)
        self.converge_checkbox.setChecked(self.converge_check)
        self.population_line_edit.setText(str(self.population_count))
        self.jump_line_edit.setText(str(self.jump_count))
        self.input_file_line_edit.setText(str(self.input_file))
        self.error_file_line_edit.setText(str(self.error_file))
        self.config_file_line_edit.setText(str(self.config_file))
        self.weights_file_line_edit.setText(str(self.weights_file))
        self.testing_obs_line_edit.setText(str(self.testing_obs_count))
        self.test_file_line_edit.setText(str(self.test_file))
        self.output_file_line_edit.setText(str(self.output_file))
        self.filepath_label.setText(f"File Path: {str(self.filepath)}")

        # Set the validators
        int_validator = QIntValidator(1, 10000) 
        self.inputs_line_edit.setValidator(int_validator)
        self.outputs_line_edit.setValidator(int_validator)
        self.hidden_nodes_line_edit.setValidator(int_validator)
        self.hidden_gen_line_edit.setValidator(int_validator)
        self.generations_line_edit.setValidator(int_validator)
        self.training_obs_line_edit.setValidator(int_validator)
        self.population_line_edit.setValidator(int_validator)
        self.jump_line_edit.setValidator(int_validator)
        self.testing_obs_line_edit.setValidator(int_validator)

        # Connect the signals and slots
        self.inputs_line_edit.textChanged.connect(self.update_inputs_count)
        self.outputs_line_edit.textChanged.connect(self.update_outputs_count)
        self.hidden_gen_line_edit.textChanged.connect(self.update_hidden_gen_count)
        self.hidden_nodes_line_edit.textChanged.connect(self.update_hidden_nodes_count)
        self.generations_line_edit.textChanged.connect(self.update_generations_count)
        self.training_obs_line_edit.textChanged.connect(self.update_training_obs_count)
        self.reduce_checkbox.stateChanged.connect(self.update_reduce_check)
        self.classification_checkbox.stateChanged.connect(self.update_classification_check)
        self.logistic_function_checkbox.stateChanged.connect(self.update_logistic_function_check)
        self.more_checkbox.stateChanged.connect(self.update_more_check)
        self.converge_checkbox.stateChanged.connect(self.update_converge_check)
        self.population_line_edit.textChanged.connect(self.update_population_count)
        self.jump_line_edit.textChanged.connect(self.update_jump_count)
        self.input_file_line_edit.textChanged.connect(self.update_input_file)
        self.error_file_line_edit.textChanged.connect(self.update_error_file)
        self.config_file_line_edit.textChanged.connect(self.update_config_file)
        self.weights_file_line_edit.textChanged.connect(self.update_weights_file)
        self.testing_obs_line_edit.textChanged.connect(self.update_testing_obs_count)
        self.test_file_line_edit.textChanged.connect(self.update_test_file)
        self.output_file_line_edit.textChanged.connect(self.update_output_file)
        self.show()
        self.output_browser.append("Welcome to the NNSOA Emulator!")
        self.output_browser.append(self.generate_parameter_report())
        self.output_browser.append("Ready to run.")
        self.output_browser.verticalScrollBar().setValue(self.output_browser.verticalScrollBar().maximum())
        
    
    def generate_parameter_report(self):
        """ Generate a report of the parameters. """
        report = ""
        report += f"Inputs: {self.inputs_count}, "
        report += f"Outputs: {self.outputs_count}, "
        report += f"Hidden Nodes: {self.hidden_nodes_count}, "
        report += f"Hidden Generations: {self.hidden_gen_count}, "
        report += f"Generations: {self.generations_count}, "
        report += f"Training Observations: {self.training_obs_count}, "
        report += f"Reduce: {self.reduce_check}, "
        report += f"Classification: {self.classification_check}, "
        report += f"Logistic Function: {self.logistic_function_check}, "
        report += f"More: {self.more_check}, "
        report += f"Converge: {self.converge_check}, "
        report += f"Population: {self.population_count}, "
        report += f"Jump: {self.jump_count}, "
        report += f"Input File: {self.input_file}, "
        report += f"Error File: {self.error_file}, "
        report += f"Config File: {self.config_file}, "
        report += f"Weights File: {self.weights_file}, "
        report += f"Testing Observations: {self.testing_obs_count}, "
        report += f"Test File: {self.test_file}, "
        report += f"Output File: {self.output_file}, "
        report += f"Filepath: {self.filepath}\n"
        return report
       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NNSOAEmulator()
    app.aboutToQuit.connect(window.on_exit)
    sys.exit(app.exec())


    