import update_path
import sys
from pathlib import Path
import PyCore
import PyDataProcess
import copy

# Update Python path for pywin32 package
if sys.platform == 'win32':
    home_path = str(Path.home())
    pywin_path = (home_path + '\\Ikomia\\Python\\lib\\site-packages\\win32')
    if pywin_path not in sys.path:
        sys.path.append(pywin_path)

    pywin_path = (home_path + '\\Ikomia\\Python\\lib\\site-packages\\win32\\lib')
    if pywin_path not in sys.path:
        sys.path.append(pywin_path)

    pywin_path = (home_path + '\\Ikomia\\Python\\lib\\site-packages\\Pythonwin')
    if pywin_path not in sys.path:
        sys.path.append(pywin_path)

# Your imports below
import os
import numpy as np
import random
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from TridentNet_git.tridentnet import add_tridentnet_config

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class TridentNetParam(PyCore.CProtocolTaskParam):

    def __init__(self):
        PyCore.CProtocolTaskParam.__init__(self)
        self.cuda = True
        self.proba = 0.8

    def setParamMap(self, paramMap):
        self.cuda = int(paramMap["cuda"])
        self.proba = int(paramMap["proba"])

    def getParamMap(self):
        paramMap = PyCore.ParamMap()
        paramMap["cuda"] = str(self.cuda)
        paramMap["proba"] = str(self.proba)
        return paramMap

# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class TridentNetProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)
        if param is None:
            self.setParam(TridentNetParam())
        else:
            self.setParam(copy.deepcopy(param))

        # get and set config model
        self.folder = os.path.dirname(os.path.realpath(__file__)) 
        self.MODEL_NAME_CONFIG = "tridentnet_fast_R_101_C4_3x"
        self.MODEL_NAME = "model_final_164568"
        self.cfg = get_cfg()
        add_tridentnet_config(self.cfg)
        self.cfg.merge_from_file(self.folder + "/TridentNet_git/configs/"+self.MODEL_NAME_CONFIG+".yaml") 
        self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pkl"
        self.loaded = False
        self.deviceFrom = ""

        # add output
        self.addOutput(PyCore.CGraphicsOutput())

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        self.beginTaskRun()

        # we use seed to keep the same color for our masks + boxes + labels (same random each time)
        random.seed(30)
        
        # Get input :
        input = self.getInput(0)
        srcImage = input.getImage()

        # Get output :
        output_image = self.getOutput(0)
        output_graph = self.getOutput(1)
        output_graph.setNewLayer("TridentNet")

        # Get parameters :
        param = self.getParam()

        # predictor
        if not self.loaded:
            print("Chargement du modèle")
            if param.cuda == False:
                self.cfg.MODEL.DEVICE = "cpu"
                self.deviceFrom = "cpu"
            else:
                self.deviceFrom = "gpu"
            self.loaded = True
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA check and load without CUDA 
        elif self.deviceFrom == "cpu" and param.cuda == True:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            add_tridentnet_config(self.cfg)
            self.cfg.merge_from_file(self.folder + "/TridentNet_git/configs/"+self.MODEL_NAME_CONFIG+".yaml") 
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pkl"
            self.deviceFrom = "gpu"
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and param.cuda == False:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.DEVICE = "cpu"
            add_tridentnet_config(self.cfg)
            self.cfg.merge_from_file(self.folder + "/TridentNet_git/configs/"+self.MODEL_NAME_CONFIG+".yaml") 
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pkl"   
            self.deviceFrom = "cpu"
            self.predictor = DefaultPredictor(self.cfg)
        
        outputs = self.predictor(srcImage)

        # get outputs instances
        output_image.setImage(srcImage)
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes

        # to numpy
        if param.cuda :
            boxes_np = boxes.tensor.cpu().numpy()
            scores_np = scores.cpu().numpy()
            classes_np = classes.cpu().numpy()
        else :
            boxes_np = boxes.tensor.numpy()
            scores_np = scores.numpy()
            classes_np = classes.numpy()

        self.emitStepProgress()
        
        # keep only the results with proba > threshold
        scores_np_tresh = list()
        for s in scores_np:
            if float(s) > param.proba:
                scores_np_tresh.append(s)
        self.emitStepProgress()

        if len(scores_np_tresh) > 0:
            # text label with score
            labels = None
            class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
            if classes is not None and class_names is not None and len(class_names) > 1:
                labels = [class_names[i] for i in classes]
            if scores_np_tresh is not None:
                if labels is None:
                    labels = ["{:.0f}%".format(s * 100) for s in scores_np_tresh]
                else:
                    labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores_np_tresh)]

            # Show Boxes + labels 
            for i in range(len(scores_np_tresh)):
                color = [random.randint(0,255), random.randint(0,255), random.randint(0,255), 255]
                properties_text = PyCore.GraphicsTextProperty()
                properties_text.color = color
                properties_text.font_size = 7
                properties_rect = PyCore.GraphicsRectProperty()
                properties_rect.pen_color = color
                output_graph.addRectangle(float(boxes_np[i][0]), float(boxes_np[i][1]), float(boxes_np[i][2] - boxes_np[i][0]), float(boxes_np[i][3] - boxes_np[i][1]), properties_rect)
                output_graph.addText(labels[i],float(boxes_np[i][0]), float(boxes_np[i][1]),properties_text)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class TridentNetProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "TridentNet"
        self.info.shortDescription = "Detectron2 project : TridentNet"
        self.info.description = "Use of Detectron2 for TridentNet model : Objects detection"
        self.info.authors = "Facebook Research"
        self.info.path = "Plugins/Python/Detectron2"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.version = "1.0.0"
        self.info.repo = "https://github.com/Ikomia-dev/IkomiaPluginsPython"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.iconPath = "icons/detectron2.png"
        self.info.keywords = "TridentNet,rcnn,detectron2,detection,objects"


    def create(self, param=None):
        # Create process object
        return TridentNetProcess(self.info.name, param)
