import PyCore
import PyDataProcess
import copy

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
        
    def setParamMap(self, paramMap):
        self.cuda = int(paramMap["cuda"])

    def getParamMap(self):
        paramMap = PyCore.ParamMap()
        paramMap["cuda"] = str(self.cuda)
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
        self.MODEL_NAME = "attn_R_50"
        self.cfg = get_cfg()
        self.folder = os.path.dirname(os.path.realpath(__file__)) 
        self.cfg.merge_from_file(self.folder + "/AdelaiDet_git/configs/BAText/TotalText/"+self.MODEL_NAME+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
        self.cfg.MODEL.WEIGHTS = self.folder + "/models/attn_R_50.pth"
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
            self.cfg = get_cfg()
            if param.cuda == False:
                self.cfg.MODEL.DEVICE = "cpu"
                self.deviceFrom = "cpu"
            else:
                self.deviceFrom = "gpu"
            self.cfg.merge_from_file(self.folder + "/AdelaiDet_git/configs/BAText/TotalText/"+self.MODEL_NAME+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/attn_R_50.pth"
            self.loaded = True
        # reload model if CUDA check and load without CUDA 
        elif self.deviceFrom == "cpu" and param.cuda == True:
            print("Chargement du modèle")
            self.deviceFrom = "gpu"
            self.cfg = get_cfg()
            self.cfg.merge_from_file(self.folder + "/AdelaiDet_git/configs/BAText/TotalText/"+self.MODEL_NAME+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/attn_R_50.pth"
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and param.cuda == False:
            print("Chargement du modèle")
            self.deviceFrom = "cpu"
            self.cfg = get_cfg()
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.merge_from_file(self.folder + "/AdelaiDet_git/configs/BAText/TotalText/"+self.MODEL_NAME+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/attn_R_50.pth"

        predictor = DefaultPredictor(self.cfg)
        predictions = predictor(srcImage)["instances"]
        self.emitStepProgress()

        # results 
        beziers = predictions.beziers.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs
        properties_text = PyCore.GraphicsTextProperty()
        properties_text.color = [0,0,0]
        values = list()
        labels = list()
        
        # draw graph + add values to output_table
        for bezier, rec, score in zip(beziers, recs, scores):
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
        self.info.path = "Plugins/Python/Detectron2/ABC_Net"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.version = "1.0.0"
        self.info.repo = "https://github.com/Ikomia-dev/IkomiaPluginsPython"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.iconPath = ""
        self.info.keywords = "Text detector,Detectron2,AdelaiDet,ABC_Net"


    def create(self, param=None):
        # Create process object
        return ABC_NetProcess(self.info.name, param)
