import PyCore
import PyDataProcess
import QtConversion
import Detectron2_FasterRCNN_process as processMod

#PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class Detectron2_FasterRCNNWidget(PyCore.CProtocolTaskWidget):

    def __init__(self, param, parent):
        PyCore.CProtocolTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = processMod.Detectron2_FasterRCNNParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # cuda parameter
        cuda_label = QLabel("CUDA")
        self.cuda_ckeck = QCheckBox()
        self.cuda_ckeck.setChecked(True)

        self.gridLayout.setColumnStretch(0,0)
        self.gridLayout.addWidget(self.cuda_ckeck, 0, 0)
        self.gridLayout.setColumnStretch(1,1)
        self.gridLayout.addWidget(cuda_label, 0, 1)
        self.gridLayout.setColumnStretch(2,2)

        # Set widget layout
        layoutPtr = QtConversion.PyQtToQt(self.gridLayout)
        self.setLayout(layoutPtr)

        if self.parameters.cuda == False:
           self.cuda_ckeck.setChecked(False)

    def onApply(self):
        # Apply button clicked slot
        if self.cuda_ckeck.isChecked():
            self.parameters.cuda = True
        else:
            self.parameters.cuda = False

        self.emitApply(self.parameters)


#--------------------
#- Factory class to build process widget object
#- Inherits PyDataProcess.CWidgetFactory from Ikomia API
#--------------------
class Detectron2_FasterRCNNWidgetFactory(PyDataProcess.CWidgetFactory):

    def __init__(self):
        PyDataProcess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "Detectron2_FasterRCNN"

    def create(self, param):
        # Create widget object
        return Detectron2_FasterRCNNWidget(param, None)
