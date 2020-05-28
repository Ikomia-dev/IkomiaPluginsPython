import sys
import numpy
import PyCore
import PyDataProcess
import Detectron2_DensePose_process as processMod
import Detectron2_DensePose_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_DensePose(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)


    def getProcessFactory(self):
        # Instantiate process object
        return processMod.Detectron2_DensePoseProcessFactory()


    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.Detectron2_DensePoseWidgetFactory()
