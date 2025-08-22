import os
import numpy as np
import cv2
import onnxruntime as ort
#from ModelLoadingHelper import DeeplabWithUpsample #this may necessary for loading pth models
from GUI_Mananger import ExecuteGUI, bmt
from example.Image_Classification_Implementaion import SubmitterClassificationImplementation
from example.Image_Segmentation_Implementation import SubmitterSegmentationImplementation
from example.ObjectDetection_Implementation import SubmitterObjectDetectionImplementation
from example.LLM_Implementation import SubmitterLLMImplementation
if __name__ == "__main__":
    # Single-Task Example
    interface = SubmitterClassificationImplementation() #Ok
    #interface = SubmitterObjectDetectionImplementation() #Ok
    #interface = SubmitterSegmentationImplementation() #Ok
    #interface = SubmitterLLMImplementation() #Ok
    ExecuteGUI(interface)

    # Multi-Domain Task Example (NG at this point !!)
    # interface1 = SubmitterClassificationImplementation()
    # interface2 = SubmitterObjectDetectionImplementation()
    # interface3 = SubmitterSegmentationImplementation()
    # ExecuteGUI([interface1, interface2, interface3])