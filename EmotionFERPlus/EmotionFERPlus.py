import PyDataProcess
import EmotionFERPlus_process as processMod
import EmotionFERPlus_widget as widgetMod


#--------------------
#- Interface class to integrate the process with Imageez application
#- Inherits PyDataProcess.CPluginProcessInterface from Imageez API
#--------------------
class EmotionFERPlus(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        #Instantiate process object
        return processMod.EmotionFERPlusProcessFactory()

    def getWidgetFactory(self):
        #Instantiate associated widget object
        return widgetMod.EmotionFERPlusWidgetFactory()
