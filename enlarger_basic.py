import numpy as np
import onnxruntime as rt
import os
import cv2

########################################## Arguments ##########################################
patch_inference = False
scale = 4
tile = 256
tile_overlap = 32
border = 0
window_size = 4
img_path = './images/madhubala.jpg'
output_path = './images/enlarged4x_madhubala.png'

################################## Loading and Preprocessing ##################################
print("Loading image...")
(imgname, imgext) = os.path.splitext(os.path.basename(img_path))
img_old = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
img_old = np.transpose(img_old if img_old.shape[2] == 1 else img_old[:, :, [2, 1, 0]], (2, 0, 1)) 
img_old = np.expand_dims(img_old, axis=0)
_, _, h_old, w_old = img_old.shape
h_pad = (h_old // window_size + 1) * window_size - h_old
w_pad = (w_old // window_size + 1) * window_size - w_old
img_old = np.concatenate([img_old, np.flip(np.copy(img_old), [2])], 2)[:, :, :h_old + h_pad, :]
img_old = np.concatenate([img_old, np.flip(np.copy(img_old), [3])], 3)[:, :, :, :w_old + w_pad]

####################################### Load ONNX Model ########################################
print("Loading model...")
sess = rt.InferenceSession("./models/model-4x.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

########################################## Inference ###########################################
print("Enlarging...")
if patch_inference is True:
    output = sess.run([label_name], {input_name: img_old})[0]
else:
    b, c, h, w = img_old.shape
    tile = min(tile, h, w)
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    sf = scale

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = np.zeros((b, c, h*sf, w*sf), dtype=np.float32)
    W = np.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_old[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = sess.run([label_name], {input_name: in_patch})[0]
            out_patch_mask = np.ones_like(out_patch)

            E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch
            W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch_mask
    output = E / W + 0.00001

################################## Postprocessing and Saving ###################################
print("Post processing...")
output = np.squeeze(output)
output = np.clip(output, 0, 1)
if output.ndim == 3:
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) 
output = (output * 255.0).round().astype(np.uint8)
print("Done! Saving...")

cv2.imwrite(output_path, output)
print("Saved.")