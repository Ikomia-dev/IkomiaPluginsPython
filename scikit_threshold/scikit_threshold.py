import PyCore
import PyDataProcess
import scikit_threshold_process as processMod
import scikit_threshold_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class scikit_threshold(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)


    def getProcessFactory(self):
        # Instantiate process object
        return processMod.scikit_thresholdProcessFactory()


    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.scikit_thresholdWidgetFactory()
