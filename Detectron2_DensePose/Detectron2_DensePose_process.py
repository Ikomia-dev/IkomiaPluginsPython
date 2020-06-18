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
from matplotlib import pyplot as plt

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_DensePoseParam(PyCore.CProtocolTaskParam):

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
        self.MODEL_NAME_CONFIG = "densepose_rcnn_R_50_FPN_s1x"
        self.MODEL_NAME = "model_final_162be9"

        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.merge_from_file(self.folder + "/DensePose_git/configs/"+self.MODEL_NAME_CONFIG+".yaml") # load densepose_rcnn_R_101_FPN_d config from file(.yaml)
        self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pkl"   # load densepose_rcnn_R_101_FPN_d config from file(.pkl)
        self.loaded = False
        self.deviceFrom = ""
        
        # add output graph
        self.addOutput(PyCore.CGraphicsOutput())

    def getProgressSteps(self, eltCount=1):
        return 2

    def run(self):
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
            self.loaded = True
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA check and load without CUDA 
        elif self.deviceFrom == "cpu" and param.cuda == True:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            add_densepose_config(self.cfg)
            self.cfg.merge_from_file(self.folder + "/DensePose_git/configs/"+self.MODEL_NAME_CONFIG+".yaml") 
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pkl"   
            self.deviceFrom = "gpu"
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and param.cuda == False:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.DEVICE = "cpu"
            add_densepose_config(self.cfg)
            self.cfg.merge_from_file(self.folder + "/DensePose_git/configs/"+self.MODEL_NAME_CONFIG+".yaml") 
            self.cfg.MODEL.WEIGHTS = self.folder + "/models/"+self.MODEL_NAME+".pkl"   
            self.deviceFrom = "cpu"
            self.predictor = DefaultPredictor(self.cfg)

        outputs = self.predictor(srcImage)["instances"]
        scores = outputs.get("scores").cpu()
        boxes_XYXY = outputs.get("pred_boxes").tensor.cpu()
        boxes_XYWH = BoxMode.convert(boxes_XYXY, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        denseposes = outputs.get("pred_densepose").to_result(boxes_XYWH)
        
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
        properties_line = PyCore.GraphicsPolylineProperty()
        properties_line.line_size = 1
        self.emitStepProgress()
        
        for i in range(len(denseposes)):
            if scores.numpy()[i] > param.proba:
                bbox_xywh = boxes_XYWH[i]
                bbox_xyxy = boxes_XYXY[i]
                result_encoded = denseposes.results[i]
                iuv_arr = DensePoseResult.decode_png_data(*result_encoded)
                # without indice surface
                self.visualize_iuv_arr(srcImage, iuv_arr, bbox_xywh, properties_line, output_graph)
                # with indice surface
                #self.visualize_iuv_arr_indiceSurface(srcImage, iuv_arr, bbox_xyxy, output_graph)
                output_graph.addRectangle(bbox_xyxy[0].item(), bbox_xyxy[1].item(), bbox_xyxy[2].item() - bbox_xyxy[0].item(), bbox_xyxy[3].item() -  bbox_xyxy[1].item(),properties_rect)
                output_graph.addText(str(scores[i].item())[:5], float(bbox_xyxy[0].item()), float(bbox_xyxy[1].item()), properties_text)
       
        output.setImage(srcImage)
        self.emitStepProgress()
        self.endTaskRun()

    # visualize densepose contours
    def visualize_iuv_arr(self, im, iuv_arr, bbox_xywh, properties_line, output_graph):
        u = iuv_arr[1,:,:].astype(float) / 255.0
        v = iuv_arr[2,:,:].astype(float) / 255.0
        extent = (
            bbox_xywh[0],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1],
            bbox_xywh[1] + bbox_xywh[3],
        )
        quadContourSetu = plt.contour(u,self.levels, extent=extent)
        quadContourSetv = plt.contour(v,self.levels, extent=extent)

        self.visualize(quadContourSetu.collections, output_graph, properties_line)
        self.visualize(quadContourSetv.collections, output_graph, properties_line)

    def visualize(self, collections, output_graph, properties_line):
        for i in range(len(collections)):
            color = collections[i].get_colors()[0]
            properties_line.pen_color = [int(color[0]*255),int(color[1]*255),int(color[2]*255)]
            for lst_pts in collections[i].get_segments():
                for j in range(len(lst_pts)-1):
                        pts0 = PyCore.CPointF(float(lst_pts[j][0]),float(lst_pts[j][1]))
                        pts1 = PyCore.CPointF(float(lst_pts[j+1][0]),float(lst_pts[j+1][1]))
                        output_graph.addPolyline([pts0, pts1],properties_line)

    # visualize densepose contours with indice surface
    def visualize_iuv_arr_indiceSurface(self, im, iuv_arr, bbox_xyxy, output_graph):
        image = im
        patch_ids = iuv_arr[0,:,:]
        u = iuv_arr[1,:,:].astype(float) / 255.0
        v = iuv_arr[2,:,:].astype(float) / 255.0
        self.contours(image, u, patch_ids, bbox_xyxy, output_graph)
        self.contours(image, v, patch_ids, bbox_xyxy, output_graph)
    
    # calcul binary codes necessary to draw lines - value for maching square cases
    def contours(self, image, arr, patch_ids, bbox_xyxy, output_graph):
        properties_line = PyCore.GraphicsPolylineProperty()
        properties_line.line_size = 1

        for patch_id in range(1, DensePoseDataRelative.N_PART_LABELS + 1):
            mask = patch_ids == patch_id
            if not np.any(mask):
                continue

            properties_line.category = str(patch_id)
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
                mp = mask[i0:i1, j0:j1]
                bin_mask_codes = mp[:-1, :-1] + mp[1:, :-1] * 2 + mp[1:, 1:] * 4 + mp[:-1, 1:] * 8
                it = np.nditer(bin_codes, flags=["multi_index"])
                properties_line.pen_color = self.level_colors_bgr[level_id]

                while not it.finished:
                    if (it[0] != 0) and (it[0] != 15):
                        i, j = it.multi_index
                        if bin_mask_codes[i, j] != 0:
                            self.draw_line(image, arr, level, properties_line, it[0], it.multi_index, bbox_xyxy, Nw, Nh, (i0, j0), output_graph)
                    it.iternext()


    # draw all lines of maching squares results
    def draw_line(self, image, arr, v, properties_line, bin_code, multi_idx, bbox_xyxy, Nw, Nh, offset, output_graph):
        lines = self.bin_code_2_lines(arr, v, bin_code, multi_idx, Nw, Nh, offset)
        x0, y0, x1, y1 = bbox_xyxy
        w = x1 - x0
        h = y1 - y0

        for line in lines:
            x0r, y0r = line[0]
            x1r, y1r = line[1]
            pts0 = PyCore.CPointF((float)(x0 + x0r * w), (float)(y0 + y0r * h))
            pts1 = PyCore.CPointF((float)(x0 + x1r * w), (float)(y0 + y1r * h))
            output_graph.addPolyline([pts0, pts1], properties_line)

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
        self.info.shortDescription = "Use of Detectron2 DensePose model."
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
        self.info.iconPath = "icons/detectron2.png"
        self.info.keywords = "human detection,rcnn,densepose,detectron2"

    def create(self, param=None):
        # Create process object
        return Detectron2_DensePoseProcess(self.info.name, param)
