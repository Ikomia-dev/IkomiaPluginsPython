import update_path
import PyCore
import PyDataProcess
import copy

# Your imports below
import torch
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
class Detectron2_PanopticSegmentationParam(PyCore.CProtocolTaskParam):

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
class Detectron2_PanopticSegmentationProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)

        #Create parameters class
        if param is None:
            self.setParam(Detectron2_PanopticSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

        # get and set config model
        self.LINK_MODEL = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        self.cfg = get_cfg()
        self.threshold = 0.5
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL)) # load config from file(.yaml)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL) # download the model (.pkl)
        self.loaded = False
        self.deviceFrom = ""
        
        # add outputs
        self.addOutput(PyCore.CImageProcessIO())
        self.addOutput(PyCore.CGraphicsOutput())

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def run(self):
        self.beginTaskRun()

        # we use seed to keep the same color for our masks + boxes + labels (same random each time)
        random.seed(30)

        # Get input :
        input = self.getInput(0)
        srcImage = input.getImage()

        # Get output :
        output_graph = self.getOutput(2)
        output_graph.setImageIndex(1)
        output_graph.setNewLayer("PanopticSegmentation")

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
            
        outputs = self.predictor(srcImage)["panoptic_seg"]

        # get outputs of model
        mask = outputs[0]
        infos = outputs[1]

        # set mask output
        mask_output = self.getOutput(0)
        if param.cuda :
            mask_output.setImage(mask.cpu().numpy())
        else :
            mask_output.setImage(mask.numpy())

        self.emitStepProgress()

        # output visualisation
        nb_objects = len(infos)

        # create random color for masks + boxes + labels
        colors = [[0,0,0]]
        for i in range(nb_objects):
        	colors.append([random.randint(0,255), random.randint(0,255), random.randint(0,255), 255])

        # get infos classes
        scores = list()
        classesThings = list()
        classesStuffs = list()
        labelsStuffs = list()

        for info in infos:
            if info["isthing"]:
                scores.append(info['score'])
                classesThings.append(info['category_id'])
            else :
                classesStuffs.append(info['category_id'])

        # text label with score - get classe name for thing and stuff from metedata
        labelsThings = None
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
        if classesThings is not None and class_names is not None and len(class_names) > 1:
            labelsThings = [class_names[i] for i in classesThings]
        if scores is not None:
            if labelsThings is None:
                labelsThings = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labelsThings = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labelsThings, scores)]
        class_names_stuff = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("stuff_classes")
        [labelsStuffs.append(class_names_stuff[x]) for x in classesStuffs]
        labels = labelsThings + labelsStuffs
        seg_ids = torch.unique(mask).tolist()

        self.emitStepProgress()

        # create masks - use for text_pos
        masks = list()
        for sid in seg_ids:
            if param.cuda:
                mymask = (mask == sid).cpu().numpy().astype(np.bool)
            else:
                mymask = (mask == sid).numpy().astype(np.bool)
            masks.append(mymask)

        # text pos = median of mask - median is less sensitive to outliers.
        if len(masks) > len(labels): # unrecognized area - no given class for area labeled 0
	        for i in range(nb_objects):
	            properties_text = PyCore.GraphicsTextProperty()
	            properties_text.color = colors[i+1]
	            properties_text.font_size = 7
	            text_pos = np.median(masks[i+1].nonzero(), axis=1)[::-1] 
	            output_graph.addText(labels[i],text_pos[0],text_pos[1],properties_text)
        else:
            for i in range(nb_objects):
                properties_text = PyCore.GraphicsTextProperty()
                properties_text.color = colors[i+1]
                properties_text.font_size = 7
                text_pos = np.median(masks[i].nonzero(), axis=1)[::-1]
                output_graph.addText(labels[i],text_pos[0],text_pos[1],properties_text)

        # output mask apply to our original image 
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
class Detectron2_PanopticSegmentationProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        self.info.name = "Detectron2_PanopticSegmentation"
        self.info.shortDescription = "Use of Detectron2 Panoptic segmentation model."
        self.info.description = "Use of Detectron2 Panoptic segmentation model : segmentation"
        self.info.authors = "Ikomia team"
        self.info.path = "Plugins/Python/Detectron2/Detectron2_PanopticSegmentation"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.version = "1.0.0"
        self.info.repo = "https://github.com/Ikomia-dev/IkomiaPluginsPython"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.iconPath = ""
        self.info.keywords = "instance segmentation, semantic segmentation, panoptic, panoptic segmentation,detectron2,segmentation"

    def create(self, param=None):
        # Create process object
        return Detectron2_PanopticSegmentationProcess(self.info.name, param)
