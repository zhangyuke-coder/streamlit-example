import cv2
import sys
import tritonclient.grpc as grpclient
import numpy as np
from in_out import INPUT_NAMES, OUTPUT_NAMES 
class image_detect():
    def __init__ (self, trition_client, img, width=640, height=640):
        print("Running in 'image' mode")
        self.img = img
        self.client = trition_client
        self.inputs = []
        self.outputs = []
        self.width = width
        self.height = height
    def process(self,):
        self.inputs.append(grpclient.InferInput(INPUT_NAMES[0], [1, 3, self.width, self.height], "FP32"))
        self.inputs[0].set_data_from_numpy(np.ones(shape=(1, 3, self.width, self.height), dtype=np.float32))
        self.outputs.append(grpclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        self.outputs.append(grpclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        self.outputs.append(grpclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        self.outputs.append(grpclient.InferRequestedOutput(OUTPUT_NAMES[3]))
        print("Creating buffer from image file...")
        
        
        # input_image = cv2.imread(str(self.img))
        input_image = np.asarray(self.img)
        if input_image is None:
            print(f"FAILED: could not load input image{str(self.img)}")
            sys.exit(1)
         
        result = self.client.infer(model_name='yolov7',
                                   inputs=self.inputs,
                                   outputs=self.outputs,
                                   client_timeout=None)
        print('success')