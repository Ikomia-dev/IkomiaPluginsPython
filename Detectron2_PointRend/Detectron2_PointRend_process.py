import update_path
import PyCore
import PyDataProcess
import copy

# Your imports below
import numpy as np
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import BoxMode
from PointRend_git.point_rend.config import add_pointrend_config
import os
import random

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_PointRendParam(PyCore.CProtocolTaskParam):

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
class Detectron2_PointRendProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)

        #Create parameters class
        if param is None:
            self.setParam(Detectron2_PointRendParam())
        else:
            self.setParam(copy.deepcopy(param))

        self.threshold = 0.5
        self.path_to_config = "/PointRend_git/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
        self.path_to_model = "/models/model_final_3c3198.pkl"
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.cfg = get_cfg()
        add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(self.folder + self.path_to_config)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.MODEL.WEIGHTS = self.folder + self.path_to_model
        self.loaded = False
        self.deviceFrom = ""

        # add output + set data type
        self.setOutputDataType(PyCore.TaskIOData.IMAGE_LABEL, 0)
        self.addOutput(PyCore.CImageProcessIO(PyCore.TaskIOData.IMAGE))
        self.addOutput(PyCore.CGraphicsOutput())

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1


    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        input = self.getInput(0)
        srcImage = input.getImage()

        # Get output :
        mask_output = self.getOutput(0)
        output_graph = self.getOutput(2)
        output_graph.setImageIndex(1)

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
        # reload model if CUDA check and load without CUDA 
        elif self.deviceFrom == "cpu" and param.cuda == True:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            add_pointrend_config(self.cfg)
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(self.folder + self.path_to_config)
            self.cfg.MODEL.WEIGHTS = self.folder + self.path_to_model
            self.deviceFrom = "gpu"
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and param.cuda == False:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            add_pointrend_config(self.cfg)
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(self.folder + self.path_to_config)
            self.cfg.MODEL.WEIGHTS = self.folder + self.path_to_model
            self.deviceFrom = "cpu"
    

        self.predictor = DefaultPredictor(self.cfg)
        outputs = self.predictor(srcImage)
        

        # get outputs instances
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes
        masks = outputs["instances"].pred_masks

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

        # create random color for masks + boxes + labels
        colors = [[0,0,0]]
        for i in range(len(boxes_np)):
        	colors.append([random.randint(0,255), random.randint(0,255), random.randint(0,255), 255])

        # text labels with scores
        labels = None
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        # Show boxes + labels
        for i in range(len(boxes_np)):
            properties_text = PyCore.GraphicsTextProperty() 
            properties_text.color = colors[i+1] # start with i+1 we don't use the first color dedicated for the label mask
            properties_text.font_size = 7
            properties_rect = PyCore.GraphicsRectProperty()
            properties_rect.pen_color = colors[i+1]
            output_graph.addRectangle(float(boxes_np[i][0]), float(boxes_np[i][1]), float(boxes_np[i][2] - boxes_np[i][0]), float(boxes_np[i][3] - boxes_np[i][1]), properties_rect)
            output_graph.addText(labels[i],float(boxes_np[i][0]), float(boxes_np[i][1]),properties_text)
        
        self.emitStepProgress()
        
        # label mask
        nb_objects = len(masks) 
        if nb_objects > 0:
            masks = masks[:nb_objects, :, :, None]
            mask_or = masks[0]*nb_objects
            for j in range(1, nb_objects):
                mask_or = torch.max(mask_or, masks[j] * (nb_objects-j))
            mask_numpy = mask_or.byte().cpu().numpy()
            mask_output.setImage(mask_numpy)

            # output mask apply to our original image 
            # inverse colors to match boxes colors
            c = colors[1:]
            c = c[::-1]
            colors = [[0,0,0]]
            for col in c:
                colors.append(col)
            self.setOutputColorMap(1, 0, colors)
        
        self.forwardInputImage(0, 1)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class Detectron2_PointRendProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Detectron2_PointRend"
        self.info.shortDescription = "Use of Detectron2 PointRend model."
        self.info.description = "Use of Detectron2 PointRend model : Segmentation mask, more precise than mask R-CNN"
        self.info.authors = "Ikomia team"
        self.info.path = "Plugins/Python/Detectron2/Detectron2_PointRend"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.version = "1.0.0"
        self.info.repo = "https://github.com/Ikomia-dev/IkomiaPluginsPython"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.iconPath = ""
        self.info.keywords = "mask,rcnn,PointRend,detectron2,segmentation"

    def create(self, param=None):
        # Create process object
        return Detectron2_PointRendProcess(self.info.name, param)
