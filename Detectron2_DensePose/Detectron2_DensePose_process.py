import PyCore
import PyDataProcess
import copy

# Your imports below
import numpy as np
import cv2
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import BoxMode
import os
from DensePose_git.densepose.data.structures import DensePoseResult
from DensePose_git.densepose.config import add_densepose_config
from DensePose_git.densepose.data.structures import DensePoseDataRelative

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_DensePoseParam(PyCore.CProtocolTaskParam):

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
class Detectron2_DensePoseProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)

        #Create parameters class
        if param is None:
            self.setParam(Detectron2_DensePoseParam())
        else:
            self.setParam(copy.deepcopy(param))
        
        # get and set config model
        self.folder = os.path.dirname(os.path.realpath(__file__)) 
        self.MODEL_NAME = "/densepose_rcnn_R_101_FPN_s1x"
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.merge_from_file(self.folder + "/DensePose_git/configs"+self.MODEL_NAME+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
        self.cfg.MODEL.WEIGHTS = self.folder + "/models" + self.MODEL_NAME + ".pkl"   # load densepose_rcnn_R_101_FPN_d config from file(.pkl)
        self.loaded = False
        self.deviceFrom = ""
        
        # add output graph
        self.addOutput(PyCore.CGraphicsOutput())

    def getProgressSteps(self, eltCount=1):
        return 2

    def run(self):
        global output_graph
        self.beginTaskRun()

        # Get input :
        input = self.getInput(0)

        # Get output :
        output = self.getOutput(0)
        output_graph = self.getOutput(1)
        output_graph.setNewLayer("DensePose")
        srcImage = input.getImage()

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
            add_densepose_config(self.cfg)
            self.cfg.merge_from_file(self.models_folder + self.MODEL_MODEL + ".yaml")
            self.cfg.MODEL.WEIGHTS = self.models_folder + self.MODEL_MODEL + ".pkl"
            self.predictor = DefaultPredictor(self.cfg)
            self.deviceFrom = "gpu"
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and param.cuda == False:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.DEVICE = "cpu"
            add_densepose_config(self.cfg)
            self.cfg.merge_from_file(self.models_folder + self.MODEL_MODEL + ".yaml")
            self.cfg.MODEL.WEIGHTS = self.models_folder + self.MODEL_MODEL + ".pkl"  
            self.predictor = DefaultPredictor(self.cfg)
            self.deviceFrom = "cpu"
        
        outputs = self.predictor(srcImage)["instances"]
        boxes_XYXY = outputs.get("pred_boxes").tensor.cpu()
        boxes_XYWH = BoxMode.convert(boxes_XYXY, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        denseposes = outputs.get("pred_densepose").to_result(boxes_XYWH)
        scores = outputs.get("scores").cpu()

        # Number of iso values betwen 0 and 1
        self.levels = np.linspace(0, 1, 9)
        cmap = cv2.COLORMAP_PARULA
        img_colors_bgr = cv2.applyColorMap((self.levels * 255).astype(np.uint8), cmap)
        self.level_colors_bgr = [
            [int(v) for v in img_color_bgr.ravel()] for img_color_bgr in img_colors_bgr
        ]

        # text and rect graph properties
        properties_text = PyCore.GraphicsTextProperty()
        properties_text.color = [255,255,255]
        properties_text.font_size = 10
        properties_rect = PyCore.GraphicsRectProperty()
        properties_rect.pen_color = [11,130,41]
        self.emitStepProgress()

        for i in range(len(denseposes)):
            bbox_xyxy = boxes_XYXY[i]
            result_encoded = denseposes.results[i]
            score = str(scores[i].item())[:5]
            iuv_arr = DensePoseResult.decode_png_data(*result_encoded)
            # densepose contours 
            self.visualize_iuv_arr(srcImage, iuv_arr, bbox_xyxy)
            # label + boxe
            if (float(score) > 0.7):
                output_graph.addRectangle(bbox_xyxy[0].item(), bbox_xyxy[1].item(), bbox_xyxy[2].item() - bbox_xyxy[0].item(), bbox_xyxy[3].item() -  bbox_xyxy[1].item(),properties_rect)
                output_graph.addText(str(score), float(bbox_xyxy[0].item()), float(bbox_xyxy[1].item()), properties_text)
        
        output.setImage(srcImage)
        self.emitStepProgress()
        self.endTaskRun()


    # visualize densepose contours with iuv array
    def visualize_iuv_arr(self, im, iuv_arr, bbox_xyxy):
        image = im
        patch_ids = iuv_arr[0,:,:]
        u = iuv_arr[1,:,:].astype(float) / 255.0
        v = iuv_arr[2,:,:].astype(float) / 255.0
        self.contours(image, u, patch_ids, bbox_xyxy)
        self.contours(image, v, patch_ids, bbox_xyxy)
    
    # calcul binary codes necessary to draw lines - value for maching square cases
    def contours(self, image, arr, patch_ids, bbox_xyxy):
        for patch_id in range(1, DensePoseDataRelative.N_PART_LABELS + 1):
            mask = patch_ids == patch_id
            if not np.any(mask):
                continue
            arr_min = np.amin(arr[mask])
            arr_max = np.amax(arr[mask])
            I, J = np.nonzero(mask)
            i0 = np.amin(I)
            i1 = np.amax(I) + 1
            j0 = np.amin(J)
            j1 = np.amax(J) + 1
            if (j1 == j0 + 1) or (i1 == i0 + 1):
                continue
            Nw = arr.shape[1] - 1
            Nh = arr.shape[0] - 1
            for level_id, level in enumerate(self.levels):
                if (level < arr_min) or (level > arr_max):
                    continue
                vp = arr[i0:i1, j0:j1] >= level
                bin_codes = vp[:-1, :-1] + vp[1:, :-1] * 2 + vp[1:, 1:] * 4 + vp[:-1, 1:] * 8
                it = np.nditer(bin_codes, flags=["multi_index"])
                color_bgr = self.level_colors_bgr[level_id]
                while not it.finished:
                    if (it[0] != 0) and (it[0] != 15):
                        self.draw_line(image, patch_id, arr, level, color_bgr, it[0], it.multi_index, bbox_xyxy, Nw, Nh, (i0, j0))
                    it.iternext()


    # draw all lines of maching squares results
    def draw_line(self, image, patch_id, arr, v, color_bgr, bin_code, multi_idx, bbox_xyxy, Nw, Nh, offset):
        lines = self.bin_code_2_lines(arr, v, bin_code, multi_idx, Nw, Nh, offset)
        x0, y0, x1, y1 = bbox_xyxy
        w = x1-x0
        h = y1-y0
        x1 = x0 + w
        y1 = y0 + h
        for line in lines:
            x0r, y0r = line[0]
            x1r, y1r = line[1]
            pt0 = (int(x0 + x0r * (x1 - x0)), int(y0 + y0r * (y1 - y0)))
            pt1 = (int(x0 + x1r * (x1 - x0)), int(y0 + y1r * (y1 - y0)))
            properties_line = PyCore.GraphicsPolylineProperty()
            properties_line.pen_color = color_bgr
            properties_line.line_size = 1
            properties_line.category = str(patch_id)
            pts0 = PyCore.CPointF(float(pt0[0]), float(pt0[1]))
            pts1 = PyCore.CPointF(float(pt1[0]), float(pt1[1]))
            lst_points = [pts0, pts1]
            output_graph.addPolyline(lst_points, properties_line)

    # maching square
    def bin_code_2_lines(self, arr, v, bin_code, multi_idx, Nw, Nh, offset):
        i0, j0 = offset
        i, j = multi_idx
        i += i0
        j += j0
        v0, v1, v2, v3 = arr[i, j], arr[i + 1, j], arr[i + 1, j + 1], arr[i, j + 1]
        x0i = float(j) / Nw
        y0j = float(i) / Nh
        He = 1.0 / Nh
        We = 1.0 / Nw
        if (bin_code == 1) or (bin_code == 14):
            a = (v - v0) / (v1 - v0)
            b = (v - v0) / (v3 - v0)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + b * We, y0j)
            return [(pt1, pt2)]
        elif (bin_code == 2) or (bin_code == 13):
            a = (v - v0) / (v1 - v0)
            b = (v - v1) / (v2 - v1)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + b * We, y0j + He)
            return [(pt1, pt2)]
        elif (bin_code == 3) or (bin_code == 12):
            a = (v - v0) / (v3 - v0)
            b = (v - v1) / (v2 - v1)
            pt1 = (x0i + a * We, y0j)
            pt2 = (x0i + b * We, y0j + He)
            return [(pt1, pt2)]
        elif (bin_code == 4) or (bin_code == 11):
            a = (v - v1) / (v2 - v1)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i + a * We, y0j + He)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif (bin_code == 6) or (bin_code == 9):
            a = (v - v0) / (v1 - v0)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif (bin_code == 7) or (bin_code == 8):
            a = (v - v0) / (v3 - v0)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i + a * We, y0j)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif bin_code == 5:
            a1 = (v - v0) / (v1 - v0)
            b1 = (v - v1) / (v2 - v1)
            pt11 = (x0i, y0j + a1 * He)
            pt12 = (x0i + b1 * We, y0j + He)
            a2 = (v - v0) / (v3 - v0)
            b2 = (v - v3) / (v2 - v3)
            pt21 = (x0i + a2 * We, y0j)
            pt22 = (x0i + We, y0j + b2 * He)
            return [(pt11, pt12), (pt21, pt22)]
        elif bin_code == 10:
            a1 = (v - v0) / (v3 - v0)
            b1 = (v - v0) / (v1 - v0)
            pt11 = (x0i + a1 * We, y0j)
            pt12 = (x0i, y0j + b1 * He)
            a2 = (v - v1) / (v2 - v1)
            b2 = (v - v3) / (v2 - v3)
            pt21 = (x0i + a2 * We, y0j + He)
            pt22 = (x0i + We, y0j + b2 * He)
            return [(pt11, pt12), (pt21, pt22)]
        return []


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class Detectron2_DensePoseProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Detectron2_DensePose"
        self.info.shortDescription = "Use of Detectron2 Faster R-CNN model."
        self.info.description = "Use of Detectron2 DensePose R-CNN model: Human Detection"
        self.info.authors = "Ikomia team"
        self.info.path = "Plugins/Python/Detectron2/Detectron2_DensePose"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.version = "1.0.0"
        self.info.repo = "https://github.com/Ikomia-dev/IkomiaPluginsPython"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.iconPath = ""
        self.info.keywords = "human detection,rcnn,densepose,detectron2"

    def create(self, param=None):
        # Create process object
        return Detectron2_DensePoseProcess(self.info.name, param)
