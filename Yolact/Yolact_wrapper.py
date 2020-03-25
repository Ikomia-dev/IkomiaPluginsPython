import torch
import torch.backends.cudnn as torch_cudnn
import cv2
from collections import defaultdict
from yolact_git.yolact import Yolact
from yolact_git.utils.functions import SavePath
from yolact_git.utils.augmentations import FastBaseTransform
from yolact_git.data import cfg, set_cfg, COLORS
from yolact_git.layers.output_utils import postprocess


color_cache = defaultdict(lambda: {})


def forward(src_img, param):
    img_numpy = None
    use_cuda = torch.cuda.device_count() >= 1
    init_config(param)

    with torch.no_grad():
        if use_cuda:
            torch_cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        net = Yolact()
        net.load_weights(param.model_path)
        net.eval()

        if use_cuda:
            net = net.cuda()

        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False

        frame = None
        if use_cuda:
            frame = torch.from_numpy(src_img).cuda().float()
        else:
            frame = torch.from_numpy(src_img).float()

        batch = FastBaseTransform()(frame.unsqueeze(0))
        predictions = net(batch)
        img_numpy = manage_outputs(predictions, frame, param)

    return img_numpy


def init_config(param):
    cfg.mask_proto_debug = False
    model_path = SavePath.from_str(param.model_path)
    config_name = model_path.model_name + "_config"
    set_cfg(config_name)


# Most parts come from function prep_display from yolact_git/eval.py
# Without command line arguments
def manage_outputs(predictions, img, param):
    crop_mask = True
    display_masks = True
    display_text = True
    display_bboxes = True
    display_scores = True
    mask_numpy = None

    # Put values in range [0 - 1]
    img_gpu = img / 255.0
    h, w, _ = img.shape

    # Post-processing
    save = cfg.rescore_bbox
    cfg.rescore_bbox = True
    t = postprocess(predictions, w, h, visualize_lincomb=False, crop_masks=crop_mask, score_threshold=param.confidence)
    cfg.rescore_bbox = save

    # Copy
    idx = t[1].argsort(0, descending=True)[:param.top_k]
    if cfg.eval_mask_branch:
        # Masks are drawn on the GPU, so don't copy
        masks = t[3][idx]

    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    # Filter available detections
    num_dets_to_consider = min(param.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < param.confidence:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    class_color = False

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            # The image might come in as RGB or BRG, depending
            color = (color[2], color[1], color[0])

            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * param.mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-param.mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]

        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        # Cumulative mask
        mask_or = masks[0]
        for j in range(1, num_dets_to_consider):
            mask_or += masks[j] * (j + 1)

        # Get the numpy array of the mask
        mask_numpy = mask_or.byte().cpu().numpy()

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if num_dets_to_consider == 0:
        return img_numpy

    if display_text or display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

    return mask_numpy, img_numpy


