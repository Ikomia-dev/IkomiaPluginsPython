import PyCore
import PyDataProcess
import QtConversion
import CovidNet_process as processMod

#PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class CovidNetWidget(PyCore.CProtocolTaskWidget):

    def __init__(self, param, parent):
        PyCore.CProtocolTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = processMod.CovidNetParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        # PyQt -> Qt wrapping
        layoutPtr = QtConversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.setLayout(layoutPtr)


    def onApply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()

        # Send signal to launch the process
        self.emitApply(self.parameters)


#--------------------
#- Factory class to build process widget object
#- Inherits PyDataProcess.CWidgetFactory from Ikomia API
#--------------------
class CovidNetWidgetFactory(PyDataProcess.CWidgetFactory):

    def __init__(self):
        PyDataProcess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "CovidNet"


    def create(self, param):
        # Create widget object
        return CovidNetWidget(param, None)
