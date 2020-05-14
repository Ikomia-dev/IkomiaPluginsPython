import sys
import numpy
import PyCore
import PyDataProcess
import scikit_MorphoSnakes_process as processMod
import scikit_MorphoSnakes_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class scikit_MorphoSnakes(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.scikit_MorphoSnakesProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.scikit_MorphoSnakesWidgetFactory()
