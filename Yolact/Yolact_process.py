import PyCore
import PyDataProcess
import copy
# Your imports below
import os
import cv2
import Yolact_wrapper as yw


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class YolactParam(PyCore.CProtocolTaskParam):

    def __init__(self):
        PyCore.CProtocolTaskParam.__init__(self)
        # Place default value initialization here        
        models_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        self.model_path = models_folder + "/yolact_im700_54_800000.pth"
        self.confidence = 0.15
        self.top_k = 15
        self.mask_alpha = 0.45


    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.confidence = float(paramMap["confidence"])
        self.top_k = float(paramMap["top_k"])
        self.mask_alpha = float(paramMap["mask_alpha"])


    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        paramMap = PyCore.ParamMap()
        paramMap["confidence"] = str(self.confidence)
        paramMap["top_k"] = str(self.top_k)
        paramMap["mask_alpha"] = str(self.mask_alpha)
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class YolactProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)
        
        #Add input/output of the process here
        self.setOutputDataType(PyCore.TaskIOData.IMAGE_LABEL, 0)
        self.addOutput(PyCore.CImageProcessIO(PyCore.TaskIOData.IMAGE))

        self.net = None
        self.class_names = []

        #Create parameters class
        if param is None:
            self.setParam(YolactParam())
        else:
            self.setParam(copy.deepcopy(param))

        # Load class names
        model_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        with open(model_folder + "/Coco_names.txt") as f:
            for row in f:
                self.class_names.append(row[:-1])


    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2


    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        img_input = self.getInput(0)
        src_img = img_input.getImage()

        # Get parameters :
        param = self.getParam()

        # Inference
        mask, dst_img = yw.forward(src_img, param)

        # Step progress bar:
        self.emitStepProgress()

        # Get mask output :
        mask_output = self.getOutput(0)
        mask_output.setImage(mask)

        # Get image output :
        img_output = self.getOutput(1)
        img_output.setImage(dst_img)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class YolactProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Yolact"
        self.info.description = "your description"
        self.info.authors = "Plugin authors"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        # self.info.iconPath = "your path to a specific icon"
        # self.info.keywords = "your keywords" -> for search


    def create(self, param=None):
        # Create process object
        return YolactProcess(self.info.name, param)
