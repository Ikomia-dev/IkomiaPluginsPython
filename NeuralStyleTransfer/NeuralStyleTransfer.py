import numpy
import PyCore
import PyDataProcess
import NeuralStyleTransfer_process as processMod
import NeuralStyleTransfer_widget as widgetMod


#--------------------
#- Interface class to integrate the process with Imageez application
#- Inherits PyDataProcess.CPluginProcessInterface from Imageez API
#--------------------
class NeuralStyleTransfer(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        #Instantiate process object
        return processMod.NeuralStyleTransferProcessFactory()

    def getWidgetFactory(self):
        #Instantiate associated widget object
        return widgetMod.NeuralStyleTransferWidgetFactory()
