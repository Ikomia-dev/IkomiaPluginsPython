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
import numpy as np
import os
import torch
from AdelaiDet_git.adet.config import get_cfg
from detectron2.engine import DefaultPredictor
from AdelaiDet_git.adet.utils.visualizer import TextVisualizer

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class ABC_NetParam(PyCore.CProtocolTaskParam):

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
class ABC_NetProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)

        #Create parameters class
        if param is None:
            self.setParam(ABC_NetParam())
        else:
            self.setParam(copy.deepcopy(param))

        # get and set config model
        self.MODEL_NAME_CONFIG = "attn_R_50"
        self.MODEL_NAME = "tt_e2e_attn_R_50"
        self.cfg = get_cfg()
        self.folder = os.path.dirname(os.path.realpath(__file__)) 
        self.cfg.merge_from_file(self.folder + "/AdelaiDet_git/configs/BAText/TotalText/"+self.MODEL_NAME_CONFIG+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
        self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pth"
        self.loaded = False
        self.deviceFrom = ""

        # add output
        self.addOutput(PyCore.CGraphicsOutput())
        self.addOutput(PyCore.CDblFeatureIO())    

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        self.beginTaskRun()

        # Get input :
        input = self.getInput(0)
        srcImage = input.getImage()

        # Get output :
        output_image = self.getOutput(0)
        output_graph = self.getOutput(1)
        output_table = self.getOutput(2)
        output_graph.setNewLayer("ABC_net")

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
            self.deviceFrom = "gpu"
            self.cfg = get_cfg()
            self.cfg.merge_from_file(self.folder + "/AdealText/"+self.MODEL_NAME_CONFIG+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pth"
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and param.cuda == False:
            print("Chargement du modèle")
            self.deviceFrom = "cpu"
            self.cfg = get_cfg()
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.merge_from_file(self.folder + "/AdelaiDet_git/configs/BAText/TotalText/"+self.MODEL_NAME_CONFIG+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pth"
            self.predictor = DefaultPredictor(self.cfg)
        
        predictions = self.predictor(srcImage)["instances"]
        self.emitStepProgress()

        # to numpy
        if param.cuda :
            beziers = predictions.beziers.cpu().numpy()
            scores = predictions.scores.cpu(),numpy()
            recs = predictions.recs.cpu().numpy()
        else :
            beziers = predictions.beziers.numpy()
            scores = predictions.scores.numpy()
            recs = predictions.recs.numpy()

        properties_text = PyCore.GraphicsTextProperty()
        properties_text.color = [0,0,0]
        values = list()
        labels = list()

        # keep only the results with proba > threshold
        scores_np_tresh = list()
        for s in scores:
            if float(s) > param.proba:
                scores_np_tresh.append(s)
        
        # draw graph + add values to output_table
        for bezier, rec, score in zip(beziers, recs, scores_np_tresh):
            polygon = TextVisualizer._bezier_to_poly(self,bezier)
            listePts = list()
            for x,y in polygon:
                listePts.append(PyCore.CPointF(float(x), float(y)))
            output_graph.addPolygon(listePts)
            
            text = TextVisualizer._decode_recognition(self,rec)
            values.append(float("{:.3f}".format(score)))
            labels.append(text)

        output_table.addValueList(values, "labels_scores", labels)
        output_image.setImage(srcImage)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class ABC_NetProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        self.info.name = "ABC_Net"
        self.info.shortDescription = "ABC_Net from AdelaiDet project, Accurate text detector using neural network"
        self.info.description = "ABC_Net from AdelaiDet project, Accurate text detector using neural network based on Detectron2"
        self.info.authors = "Ikomia team"
        self.info.path = "Plugins/Python/Detectron2"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.version = "1.0.0"
        self.info.repo = "https://github.com/Ikomia-dev/IkomiaPluginsPython"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.iconPath = "icons/detectron2.png"
        self.info.keywords = "Text detector,Detectron2,AdelaiDet,ABC_Net"


    def create(self, param=None):
        # Create process object
        return ABC_NetProcess(self.info.name, param)
