"""
# PyTroch model conversion

Reference: https://jeanvitor.com/how-to-load-pytorch-models-with-opencv/

__author__ = "P Rajan"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""
import torch
import sys
import os
import cv2
import numpy as np
import onnx
import torch
import torch.nn as nn 
import RegARegBCombined as concatAB
sys.path.append(os.path.join("..", "segment-pytorch"))
device = ("cuda" if torch.cuda.is_available() else "cpu")
# # TODO: import this from RegARegBCombined.py after the merge
# class RegARegBCombinedModel(nn.Module):
    
#     def __init__(self, original_model): 
#         super(RegARegBCombinedModel, self).__init__()
        
#         self.layer = original_model
        
#     def forward(self,x):
#         return torch.cat(self.layer(x), -1)

def convert_model_concatAB(original_model_path, new_model_name):
    
    # Loading the original model
    original_model =  torch.jit.load(original_model_path)
    
    # Initializing the new model
    new_model = concatAB.RegARegBCombinedModel(original_model).to(device)
    
    # Saving the new model
    model_script = torch.jit.script(new_model)
    
    # Saving The new model 
    new_model_path, _ = os.path.split(original_model_path)
    model_script.save(new_model_path + "/" + new_model_name)
    
def convert_pyt_segnet_onnx(model_fullpath, onnx_model_path, n, c, h, w, predict=False, image_path = None):
    """
    Converts a .pt model to .onnx. If predict, will display the prediction of the new onnx model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_fullpath, device)
    model.eval()
    x = torch.randn(n,c,h,w).to(device)
    # torch.onnx.export(model, x, onnx_model_path, verbose=False)
    torch.onnx.export(model, x, onnx_model_path, verbose=False, do_constant_folding=False) # Ref: https://glaringlee.github.io/onnx.html
    # torch.onnx.export(model, x, onnx_model_path, verbose=False,
    #                 input_names=["input"],
    #                 output_names=["output"],
    #                 dynamic_axes={
    #                     "input": {0: "batch_size", 2: "height", 3: "width"},
    #                     "output": {0: "batch_size", 2: "height", 3: "width"}
    #                 })
    if predict:
        if image_path == None:
            print("Please provide an image path to test")
            return
        image = cv2.imread(image_path)
        net = cv2.dnn.readNetFromONNX(onnx_model_path) 
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0/255, swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        preds = np.squeeze(preds)
        preds[preds<=0.7] = 0
        preds[preds>0.7] = 1
        preds = preds*255
        mask = np.zeros(np.shape(image))
        mask[:,:,0] = preds
        mask[:,:,1] = preds
        mask[:,:,2] = preds
        mask = mask.astype('uint8')
        overlay = cv2.addWeighted(image, 1, mask, .7, 0)
        cv2.imshow("Prediction For Segmentation", overlay)
        cv2.waitKey()

def convert_pyt_hub3ch_onnx(model_fullpath, onnx_model_path, predict=False, image_path=None):
    """
    Converts .pt model to .onnx. If predict will display the prediction directly from
    the model to the console.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_fullpath, device)
    model.eval()
    x = torch.randn(1,3, 256, 256).to(device)
    torch.onnx.export(model, x, onnx_model_path, verbose=False)
    if predict:
        if image_path == None:
            print("Please provide an image path")
            return
        net = cv2.dnn.readNetFromONNX(onnx_model_path)
        image = cv2.imread(image_path)
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0/255, swapRB=True, crop=False)
        net.setInput(blob)
        out_layer_names = net.getUnconnectedOutLayersNames() # get the names of the unconnected layers
        preds = net.forward(outBlobNames=out_layer_names) 
        print(preds)
        


def convert_pyt_hub4ch_onnx(model_fullpath, onnx_model_path, predict = False, image_path = None, mask_path=None):
    """
    Converts .pt model to .onnx. If predict will display the prediction directly from
    the model to the console.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = torch.jit.load(model_fullpath, device)
    model.eval()
    x = torch.randn(1,4, 256, 256).to(device)
    torch.onnx.export(model, x, onnx_model_path, verbose=False)
    if predict:
        if image_path == None:
            print("Please provide an image path")
            return
        if mask_path == None:
            print("Please provide a mask path")
            return
        net = cv2.dnn.readNetFromONNX(onnx_model_path)
        img = cv2.imread(image_path)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        msk = msk * 255
        combined_image = cv2.merge((img, msk))
        blob = cv2.dnn.blobFromImage(combined_image, scalefactor=1.0/255, swapRB=True, crop=False)
        net.setInput(blob)  # sets image input to blob
        out_layer_names = net.getUnconnectedOutLayersNames() # get the names of the unconnected layers
        preds = net.forward(outBlobNames=out_layer_names)    # forward pass
        print(preds)


# if __name__ == "__main__":
    
#     # Example usages
#     convert_pyt_hub3ch_onnx("3ChaInpute_RegARegBCombined_20231116_T150541.pt", "3ChaInpute_RegARegBCombined_20231116_T150541.onnx", True, "test_256x256.png")
#     convert_pyt_hub4ch_onnx("RegARegBCombined_20231108_T121036.pt", "RegARegBCombined_20231108_T121036.onnx", True, "test_256x256.png", "mask_test_256x256.png")
#     convert_pyt_segnet_onnx("model_cyc_1_ep_1000_bs_16_lr_1e-05_20231102_T133358.pt", "model_cyc_1_ep_1000_bs_16_lr_1e-05_20231102_T133358.onnx", 1, 3, 1280, 720, True, "test_full_image.png")

if __name__ == "__main__":
    
    # model_in_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena/models/shaft_segmentation"
    # model_name = "model_cyc_3_ep_200_bs_64_lr_1e-05_20231207_T145104.pt"
    # in_model = os.path.join(model_in_dir, model_name)

    model_in_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena/models/hub_regression"
    # model_name = "chkpt_ep512_D7_res18_lean_hub_cyc_1_ep_1000_bs_64_encoder_lr_0.0001_lr_0.001_20231220_T125056.pt"
    model_name = "model_cyc_1_ep_400_bs_64_lr_0.0001_20231220_T173517.pt"
    
    # # # NOTE: Enable this before hub model conversion to concatenate the A and B output branches
    concat = False
    if concat:
        catAB_model = "catAB_" + model_name
        convert_model_concatAB(os.path.join(model_in_dir, model_name), catAB_model)
        in_model = os.path.join(model_in_dir, catAB_model)
        model_name = catAB_model
    else:
        in_model = os.path.join(model_in_dir, model_name)

    model_out_dir = os.path.join(model_in_dir, "onnx") 
    model_name_onnx, _ = os.path.splitext(model_name)
    model_name_onnx = model_name_onnx + ".onnx"
    out_model = os.path.join(model_out_dir, model_name_onnx)

    # Example usages
    # convert_pyt_hub3ch_onnx("3ChaInpute_RegARegBCombined_20231116_T150541.pt", "3ChaInpute_RegARegBCombined_20231116_T150541.onnx", True, "test_256x256.png")
    # convert_pyt_hub4ch_onnx("RegARegBCombined_20231108_T121036.pt", "RegARegBCombined_20231108_T121036.onnx", True, "test_256x256.png", "mask_test_256x256.png")
    # convert_pyt_segnet_onnx("model_cyc_3_ep_200_bs_64_lr_0.0001_20231206_T174318.pt", "model_cyc_3_ep_200_bs_64_lr_0.0001_20231206_T174318.onnx", 1, 3, 1280, 720, True, "test_full_image.png")

    

    # convert_pyt_segnet_onnx(in_model, out_model, 1, 3, 1280, 720, True, "test_full_image.png")
    convert_pyt_hub4ch_onnx(in_model, out_model, True, "test_256x256.png", "mask_test_256x256.png")
    # convert_pyt_hub3ch_onnx(in_model, out_model, True, "test_256x256.png")
