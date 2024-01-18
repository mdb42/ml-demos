# Form implementation generated from reading ui file 'gui\nnsoa.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(587, 517)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.splitter = QtWidgets.QSplitter(parent=self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter.setObjectName("splitter")
        self.gridLayoutWidget = QtWidgets.QWidget(parent=self.splitter)
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.top_splitter_panel = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.top_splitter_panel.setContentsMargins(0, 0, 0, 0)
        self.top_splitter_panel.setObjectName("top_splitter_panel")
        self.top_layout = QtWidgets.QHBoxLayout()
        self.top_layout.setObjectName("top_layout")
        self.top_left_layout = QtWidgets.QVBoxLayout()
        self.top_left_layout.setObjectName("top_left_layout")
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.setObjectName("main_layout")
        self.hyperparametes_layout = QtWidgets.QVBoxLayout()
        self.hyperparametes_layout.setObjectName("hyperparametes_layout")
        self.inputs_layout = QtWidgets.QVBoxLayout()
        self.inputs_layout.setObjectName("inputs_layout")
        self.inputs_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.inputs_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.inputs_label.setObjectName("inputs_label")
        self.inputs_layout.addWidget(self.inputs_label)
        self.inputs_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.inputs_line_edit.setObjectName("inputs_line_edit")
        self.inputs_layout.addWidget(self.inputs_line_edit)
        self.hyperparametes_layout.addLayout(self.inputs_layout)
        self.outputs_layout = QtWidgets.QVBoxLayout()
        self.outputs_layout.setObjectName("outputs_layout")
        self.outputs_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.outputs_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.outputs_label.setObjectName("outputs_label")
        self.outputs_layout.addWidget(self.outputs_label)
        self.outputs_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.outputs_line_edit.setObjectName("outputs_line_edit")
        self.outputs_layout.addWidget(self.outputs_line_edit)
        self.hyperparametes_layout.addLayout(self.outputs_layout)
        self.hidden_nodes_layout = QtWidgets.QVBoxLayout()
        self.hidden_nodes_layout.setObjectName("hidden_nodes_layout")
        self.hidden_nodes_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.hidden_nodes_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.hidden_nodes_label.setObjectName("hidden_nodes_label")
        self.hidden_nodes_layout.addWidget(self.hidden_nodes_label)
        self.hidden_nodes_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.hidden_nodes_line_edit.setObjectName("hidden_nodes_line_edit")
        self.hidden_nodes_layout.addWidget(self.hidden_nodes_line_edit)
        self.hyperparametes_layout.addLayout(self.hidden_nodes_layout)
        self.hidden_gen_layout = QtWidgets.QVBoxLayout()
        self.hidden_gen_layout.setObjectName("hidden_gen_layout")
        self.hidden_gen_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.hidden_gen_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.hidden_gen_label.setObjectName("hidden_gen_label")
        self.hidden_gen_layout.addWidget(self.hidden_gen_label)
        self.hidden_gen_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.hidden_gen_line_edit.setObjectName("hidden_gen_line_edit")
        self.hidden_gen_layout.addWidget(self.hidden_gen_line_edit)
        self.hyperparametes_layout.addLayout(self.hidden_gen_layout)
        self.generations_layout = QtWidgets.QVBoxLayout()
        self.generations_layout.setObjectName("generations_layout")
        self.generations_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.generations_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.generations_label.setObjectName("generations_label")
        self.generations_layout.addWidget(self.generations_label)
        self.generations_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.generations_line_edit.setObjectName("generations_line_edit")
        self.generations_layout.addWidget(self.generations_line_edit)
        spacerItem = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.generations_layout.addItem(spacerItem)
        self.hyperparametes_layout.addLayout(self.generations_layout)
        self.main_layout.addLayout(self.hyperparametes_layout)
        self.top_center_layout = QtWidgets.QVBoxLayout()
        self.top_center_layout.setObjectName("top_center_layout")
        self.run_button = QtWidgets.QPushButton(parent=self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.run_button.sizePolicy().hasHeightForWidth())
        self.run_button.setSizePolicy(sizePolicy)
        self.run_button.setMinimumSize(QtCore.QSize(0, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.run_button.setFont(font)
        self.run_button.setObjectName("run_button")
        self.top_center_layout.addWidget(self.run_button)
        self.training_obs_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.training_obs_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.training_obs_label.setObjectName("training_obs_label")
        self.top_center_layout.addWidget(self.training_obs_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.training_obs_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.training_obs_line_edit.setObjectName("training_obs_line_edit")
        self.horizontalLayout.addWidget(self.training_obs_line_edit)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.top_center_layout.addLayout(self.horizontalLayout)
        self.main_checkbox_layout = QtWidgets.QFrame(parent=self.gridLayoutWidget)
        self.main_checkbox_layout.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.main_checkbox_layout.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.main_checkbox_layout.setObjectName("main_checkbox_layout")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.main_checkbox_layout)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.reduce_checkbox = QtWidgets.QCheckBox(parent=self.main_checkbox_layout)
        self.reduce_checkbox.setObjectName("reduce_checkbox")
        self.verticalLayout_5.addWidget(self.reduce_checkbox)
        self.classification_checkbox = QtWidgets.QCheckBox(parent=self.main_checkbox_layout)
        self.classification_checkbox.setObjectName("classification_checkbox")
        self.verticalLayout_5.addWidget(self.classification_checkbox)
        self.logistic_function_checkbox = QtWidgets.QCheckBox(parent=self.main_checkbox_layout)
        self.logistic_function_checkbox.setObjectName("logistic_function_checkbox")
        self.verticalLayout_5.addWidget(self.logistic_function_checkbox)
        self.top_center_layout.addWidget(self.main_checkbox_layout)
        self.aux_checkbox_layout = QtWidgets.QFrame(parent=self.gridLayoutWidget)
        self.aux_checkbox_layout.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.aux_checkbox_layout.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.aux_checkbox_layout.setObjectName("aux_checkbox_layout")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.aux_checkbox_layout)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.more_checkbox = QtWidgets.QCheckBox(parent=self.aux_checkbox_layout)
        self.more_checkbox.setObjectName("more_checkbox")
        self.verticalLayout_6.addWidget(self.more_checkbox)
        self.converge_checkbox = QtWidgets.QCheckBox(parent=self.aux_checkbox_layout)
        self.converge_checkbox.setObjectName("converge_checkbox")
        self.verticalLayout_6.addWidget(self.converge_checkbox)
        self.top_center_layout.addWidget(self.aux_checkbox_layout)
        self.ga_layout = QtWidgets.QHBoxLayout()
        self.ga_layout.setObjectName("ga_layout")
        self.population_layout = QtWidgets.QVBoxLayout()
        self.population_layout.setObjectName("population_layout")
        self.population_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.population_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.population_label.setObjectName("population_label")
        self.population_layout.addWidget(self.population_label)
        self.population_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.population_line_edit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.population_line_edit.setObjectName("population_line_edit")
        self.population_layout.addWidget(self.population_line_edit)
        self.ga_layout.addLayout(self.population_layout)
        self.jump_layout = QtWidgets.QVBoxLayout()
        self.jump_layout.setObjectName("jump_layout")
        self.jump_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.jump_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.jump_label.setObjectName("jump_label")
        self.jump_layout.addWidget(self.jump_label)
        self.jump_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.jump_line_edit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.jump_line_edit.setObjectName("jump_line_edit")
        self.jump_layout.addWidget(self.jump_line_edit)
        self.ga_layout.addLayout(self.jump_layout)
        self.top_center_layout.addLayout(self.ga_layout)
        spacerItem2 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.top_center_layout.addItem(spacerItem2)
        self.main_layout.addLayout(self.top_center_layout)
        self.top_left_layout.addLayout(self.main_layout)
        self.test_layout = QtWidgets.QHBoxLayout()
        self.test_layout.setObjectName("test_layout")
        self.test_button = QtWidgets.QPushButton(parent=self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.test_button.sizePolicy().hasHeightForWidth())
        self.test_button.setSizePolicy(sizePolicy)
        self.test_button.setMinimumSize(QtCore.QSize(0, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.test_button.setFont(font)
        self.test_button.setObjectName("test_button")
        self.test_layout.addWidget(self.test_button)
        self.testing_obs_layout = QtWidgets.QVBoxLayout()
        self.testing_obs_layout.setObjectName("testing_obs_layout")
        self.testing_obs_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.testing_obs_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.testing_obs_label.setObjectName("testing_obs_label")
        self.testing_obs_layout.addWidget(self.testing_obs_label)
        self.testing_obs_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.testing_obs_line_edit.setObjectName("testing_obs_line_edit")
        self.testing_obs_layout.addWidget(self.testing_obs_line_edit)
        self.test_layout.addLayout(self.testing_obs_layout)
        self.test_file_layout = QtWidgets.QVBoxLayout()
        self.test_file_layout.setObjectName("test_file_layout")
        self.testing_file_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.testing_file_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.testing_file_label.setObjectName("testing_file_label")
        self.test_file_layout.addWidget(self.testing_file_label)
        self.test_file_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.test_file_line_edit.setObjectName("test_file_line_edit")
        self.test_file_layout.addWidget(self.test_file_line_edit)
        self.test_layout.addLayout(self.test_file_layout)
        self.top_left_layout.addLayout(self.test_layout)
        self.top_layout.addLayout(self.top_left_layout)
        self.files_layout = QtWidgets.QVBoxLayout()
        self.files_layout.setObjectName("files_layout")
        self.input_file_layout = QtWidgets.QVBoxLayout()
        self.input_file_layout.setObjectName("input_file_layout")
        self.input_file_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.input_file_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.input_file_label.setObjectName("input_file_label")
        self.input_file_layout.addWidget(self.input_file_label)
        self.input_file_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.input_file_line_edit.setObjectName("input_file_line_edit")
        self.input_file_layout.addWidget(self.input_file_line_edit)
        self.files_layout.addLayout(self.input_file_layout)
        self.error_file_layout = QtWidgets.QVBoxLayout()
        self.error_file_layout.setObjectName("error_file_layout")
        self.error_file_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.error_file_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.error_file_label.setObjectName("error_file_label")
        self.error_file_layout.addWidget(self.error_file_label)
        self.error_file_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.error_file_line_edit.setObjectName("error_file_line_edit")
        self.error_file_layout.addWidget(self.error_file_line_edit)
        self.files_layout.addLayout(self.error_file_layout)
        self.config_file_layout = QtWidgets.QVBoxLayout()
        self.config_file_layout.setObjectName("config_file_layout")
        self.config_file_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.config_file_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.config_file_label.setObjectName("config_file_label")
        self.config_file_layout.addWidget(self.config_file_label)
        self.config_file_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.config_file_line_edit.setObjectName("config_file_line_edit")
        self.config_file_layout.addWidget(self.config_file_line_edit)
        self.files_layout.addLayout(self.config_file_layout)
        self.weights_file_layout = QtWidgets.QVBoxLayout()
        self.weights_file_layout.setObjectName("weights_file_layout")
        self.weights_file_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.weights_file_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.weights_file_label.setObjectName("weights_file_label")
        self.weights_file_layout.addWidget(self.weights_file_label)
        self.weights_file_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.weights_file_line_edit.setObjectName("weights_file_line_edit")
        self.weights_file_layout.addWidget(self.weights_file_line_edit)
        self.files_layout.addLayout(self.weights_file_layout)
        spacerItem3 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.files_layout.addItem(spacerItem3)
        self.output_file_layout = QtWidgets.QVBoxLayout()
        self.output_file_layout.setObjectName("output_file_layout")
        self.output_file_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.output_file_label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.output_file_label.setObjectName("output_file_label")
        self.output_file_layout.addWidget(self.output_file_label)
        self.output_file_line_edit = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.output_file_line_edit.setObjectName("output_file_line_edit")
        self.output_file_layout.addWidget(self.output_file_line_edit)
        self.files_layout.addLayout(self.output_file_layout)
        self.top_layout.addLayout(self.files_layout)
        self.top_splitter_panel.addLayout(self.top_layout, 0, 0, 1, 1)
        self.filepath_label = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.filepath_label.setObjectName("filepath_label")
        self.top_splitter_panel.addWidget(self.filepath_label, 1, 0, 1, 1)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(parent=self.splitter)
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.bottom_splitter_panel = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.bottom_splitter_panel.setContentsMargins(0, 0, 0, 0)
        self.bottom_splitter_panel.setObjectName("bottom_splitter_panel")
        self.output_browser = QtWidgets.QTextBrowser(parent=self.gridLayoutWidget_2)
        self.output_browser.setObjectName("output_browser")
        self.bottom_splitter_panel.addWidget(self.output_browser, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.splitter, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 587, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtGui.QAction(parent=MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.inputs_label.setText(_translate("MainWindow", "Inputs"))
        self.outputs_label.setText(_translate("MainWindow", "Outputs"))
        self.hidden_nodes_label.setText(_translate("MainWindow", "Hidden Nodes"))
        self.hidden_gen_label.setText(_translate("MainWindow", "Hidden Gen"))
        self.generations_label.setText(_translate("MainWindow", "Generations"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.training_obs_label.setText(_translate("MainWindow", "Training Obs"))
        self.reduce_checkbox.setText(_translate("MainWindow", "Reduce"))
        self.classification_checkbox.setText(_translate("MainWindow", "Classification"))
        self.logistic_function_checkbox.setText(_translate("MainWindow", "Logistic Function"))
        self.more_checkbox.setText(_translate("MainWindow", "More"))
        self.converge_checkbox.setText(_translate("MainWindow", "Converge"))
        self.population_label.setText(_translate("MainWindow", "Population"))
        self.jump_label.setText(_translate("MainWindow", "Jump"))
        self.test_button.setText(_translate("MainWindow", "Test"))
        self.testing_obs_label.setText(_translate("MainWindow", "Testing Obs"))
        self.testing_file_label.setText(_translate("MainWindow", "Test File"))
        self.input_file_label.setText(_translate("MainWindow", "Input File"))
        self.error_file_label.setText(_translate("MainWindow", "Error File"))
        self.config_file_label.setText(_translate("MainWindow", "Config File"))
        self.weights_file_label.setText(_translate("MainWindow", "Weights File"))
        self.output_file_label.setText(_translate("MainWindow", "Output File"))
        self.filepath_label.setText(_translate("MainWindow", "File Path:"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
