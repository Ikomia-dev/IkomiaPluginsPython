import update_path
import PyCore
import PyDataProcess
import copy

# Your imports below
import detectron2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import random

# --------------------
# - Class to handle the procesZs parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_FasterRCNNParam(PyCore.CProtocolTaskParam):

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
class Detectron2_FasterRCNNProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)

        #Create parameters class
        if param is None:
            self.setParam(Detectron2_FasterRCNNParam())
        else:
            self.setParam(copy.deepcopy(param))

        # get and set config model
        self.LINK_MODEL = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        self.threshold = 0.5
        self.cfg = get_cfg()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL)) # load config from file(.yaml)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL) # download the model (.pkl)
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

        # we use seed to keep the same color for our boxes + labels (same random each time)
        random.seed(30)
        
        # Get input :
        input = self.getInput(0)
        srcImage = input.getImage()

        # Get output :
        output_image = self.getOutput(0)
        output_graph = self.getOutput(1)
        output_graph.setNewLayer("FasterRCNN")

        # Get parameters :
        param = self.getParam()

        # predictor
        if not self.loaded:
            print("Chargement du modèle")
            if param.cuda == False:
                self.cfg.MODEL.DEVICE = "cpu"
                self.deviceFrom = "cpu"
            self.predictor = DefaultPredictor(self.cfg)
            self.loaded = True
        # reload model if CUDA check and load without CUDA 
        elif self.deviceFrom == "cpu" and param.cuda == True:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL)) # load config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL) # download the model (.pkl)
            self.predictor = DefaultPredictor(self.cfg)
            self.deviceFrom = "gpu"
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and param.cuda == False:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL)) # load config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL) # download the model (.pkl)
            self.predictor = DefaultPredictor(self.cfg)
            self.deviceFrom = "cpu"
            
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

        # text label with score
        labels = None
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        # Show Boxes + labels 
        for i in range(len(boxes_np)):
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
class Detectron2_FasterRCNNProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Detectron2_FasterRCNN"
        self.info.shortDescription = "Use of Detectron2 Faster R-CNN model."
        self.info.description = "Use of Detectron2 Faster R-CNN model : Objects detection"
        self.info.authors = "Ikomia team"
        self.info.path = "Plugins/Python/Detectron2/Detectron2_FasterRCNN"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.version = "1.0.0"
        self.info.repo = "https://github.com/Ikomia-dev/IkomiaPluginsPython"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.iconPath = ""
        self.info.keywords = "faster,rcnn,fasterRCNN,detectron2,detection"

    def create(self, param=None):
        # Create process object
        return Detectron2_FasterRCNNProcess(self.info.name, param)
