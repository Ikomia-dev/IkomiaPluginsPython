import sys
import numpy
import PyCore
import PyDataProcess
import Detectron2_KeyptsRCNN_process as processMod
import Detectron2_KeyptsRCNN_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_KeyptsRCNN(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.Detectron2_KeyptsRCNNProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.Detectron2_KeyptsRCNNWidgetFactory()
