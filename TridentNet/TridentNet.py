import sys
import numpy
import PyCore
import PyDataProcess
import TridentNet_process as processMod
import TridentNet_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class TridentNet(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)


    def getProcessFactory(self):
        # Instantiate process object
        return processMod.TridentNetProcessFactory()


    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.TridentNetWidgetFactory()
