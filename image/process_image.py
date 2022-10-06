import cv2
import sys
import tritonclient.grpc as grpcclient
import numpy as np
from in_out import INPUT_NAMES, OUTPUT_NAMES
from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import COCOLabels
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

        self.inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, self.width, self.height], "FP32"))
        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))
        print("Creating buffer from image file...")
        
        
        # input_image = cv2.imread(str(self.img))
        input_image = np.asarray(self.img)
        if input_image is None:
            print(f"FAILED: could not load input image{str(self.img)}")
            sys.exit(1)
        # print("input_image:{}".format(input_image.shape))
        input_image_buffer = preprocess(input_image, [self.width, self.height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        self.inputs[0].set_data_from_numpy(input_image_buffer)
        
        print("Invoking inference...")
        
        results = self.client.infer(model_name = 'yolov7',
                                   inputs=self.inputs,
                                   outputs=self.outputs,
                                   client_timeout=None)


        for output in OUTPUT_NAMES:
            result = results.as_numpy(output)

            print(f"Received result buffer \"{output}\" of size {result.shape}")
            print(f"Naive buffer sum: {np.sum(result)}")
            
            
        num_dets = results.as_numpy(OUTPUT_NAMES[0])
        det_boxes = results.as_numpy(OUTPUT_NAMES[1])
        det_scores = results.as_numpy(OUTPUT_NAMES[2])
        det_classes = results.as_numpy(OUTPUT_NAMES[3])  
               
        detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, input_image.shape[1], input_image.shape[0], [self.width, self.height])
        sum = 0
        for box in detected_objects:
            sum = sum + 1
            print(f"{COCOLabels(box.classID).name}: {box.confidence}")
            input_image = render_box(input_image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
            size = get_text_size(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
            input_image = render_filled_box(input_image, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
            
            input_image = render_text(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)
        print(sum)
        return input_image
