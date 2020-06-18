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
import detectron2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import random

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_KeyptsRCNNParam(PyCore.CProtocolTaskParam):

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
class Detectron2_KeyptsRCNNProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)
        if param is None:
            self.setParam(Detectron2_KeyptsRCNNParam())
        else:
            self.setParam(copy.deepcopy(param))

        # get and set config model
        self.LINK_MODEL = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
        self.cfg = get_cfg()
        self.threshold = 0.5
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL)) # load config from file(.yaml)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL) # download the model (.pkl)
        self.loaded = False
        self.deviceFrom = ""
        
        # add output
        self.addOutput(PyCore.CGraphicsOutput())
        
        # keypoint threshold
        self._KEYPOINT_THRESHOLD = 0.05

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def run(self):

        self.beginTaskRun()

        # we use seed to keep the same color for our boxes + labels (same random each time)
        random.seed(30)

        # Get input :
        input = self.getInput(0)
        srcImage = input.getImage()

        # Get output :
        output_graph = self.getOutput(1)
        output_graph.setNewLayer("KeypointRCNN")

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
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes
        keypoints = outputs["instances"].pred_keypoints

        # to numpy
        if param.cuda :
            boxes_np = boxes.tensor.cpu().numpy()
            scores_np = scores.cpu().numpy()
            classes_np = classes.cpu().numpy()
            keypoints_np = keypoints.cpu().numpy()
        else :
            boxes_np = boxes.tensor.numpy()
            scores_np = scores.numpy()
            classes_np = classes.numpy()
            keypoints_np = keypoints.numpy()
        
        self.emitStepProgress()

        # keep only the results with proba > threshold
        scores_np_tresh = list()
        for s in scores_np:
            if float(s) > param.proba:
                scores_np_tresh.append(s)
        
        if len(scores_np_tresh) > 0:
            # create random color for boxes and labels
            colors = []
            for i in range(len(scores_np_tresh)):
                colors.append([random.randint(0,255), random.randint(0,255), random.randint(0,255), 255])

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

            # Show boxes + labels
            for i in range(len(scores_np_tresh)):
                properties_text = PyCore.GraphicsTextProperty()
                properties_text.color = colors[i] # start with i+1 we don't use the first color dedicated for the label mask
                properties_text.font_size = 7
                properties_rect = PyCore.GraphicsRectProperty()
                properties_rect.pen_color = colors[i]
                output_graph.addRectangle(float(boxes_np[i][0]), float(boxes_np[i][1]), float(boxes_np[i][2] - boxes_np[i][0]), float(boxes_np[i][3] - boxes_np[i][1]), properties_rect)
                output_graph.addText(labels[i],float(boxes_np[i][0]), float(boxes_np[i][1]),properties_text)
        
            self.emitStepProgress()

            # keypoints
            properties_point = PyCore.GraphicsPointProperty()
            properties_point.pen_color = [0, 0, 0, 255]
            properties_point.brush_color = [0, 0, 255, 255]
            properties_point.size = 10

            # get keypoints name if prob > Threshold
            keypoint_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_names")
            for keypoints_obj in keypoints_np[:len(scores_np_tresh)]:
                visible_keypoints = {}
                for idx, kp in enumerate(keypoints_obj):
                    x, y, prob = kp
                    if prob > self._KEYPOINT_THRESHOLD:
                        pts = PyCore.CPointF(float(x), float(y))
                        output_graph.addPoint(pts, properties_point)
                        if keypoint_names:
                            keypoint_name = keypoint_names[idx]
                            visible_keypoints[keypoint_name] = (x, y)
                
                # keypoints connections
                if MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_connection_rules"):
                    for kpName_0, kpName_1, color in MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_connection_rules"):
                        for kpName_0, kpName_1, color in MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_connection_rules"):
                            if kpName_0 in visible_keypoints and kpName_1 in visible_keypoints:
                                x0, y0 = visible_keypoints[kpName_0]
                                x1, y1 = visible_keypoints[kpName_1]
                                color = [x for x in color]
                                color.append(255)
                                properties_line = PyCore.GraphicsPolylineProperty()
                                properties_line.pen_color = color	
                                pts0 = PyCore.CPointF(float(x0), float(y0))
                                pts1 = PyCore.CPointF(float(x1), float(y1))
                                lst_points = [pts0, pts1]
                                output_graph.addPolyline(lst_points, properties_line)
        else:
            self.emitStepProgress()
            
		# Set input image to output 0 
        self.forwardInputImage(0, 0)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class Detectron2_KeyptsRCNNProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        self.info.name = "Detectron2_KeyptsRCNN"
        self.info.shortDescription = "Use of Detectron2 Keypoint detection model."
        self.info.description = "Use of Detectron2 Keypoint detection model : Keypoints R-CNN, human detection"
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
        self.info.keywords = "human detection,keypoint,detectron2,human skeleton"

    def create(self, param=None):
        # Create process object
        return Detectron2_KeyptsRCNNProcess(self.info.name, param)
