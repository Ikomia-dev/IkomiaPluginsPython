import PyCore
import PyDataProcess
import ColorfulImageColorization_process as processMod
import ColorfulImageColorization_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class ColorfulImageColorization(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.ColorfulImageColorizationProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.ColorfulImageColorizationWidgetFactory()
