from tkinter import image_types
from numpy import imag
from image.process_image import image_detect
import trition
import tritonclient.grpc as grpclient
from PIL import Image
image = Image.open('./dog.jpg')
client = trition.init_triton('127.0.0.1:8001')

image_detect = image_detect(client, image)
image_detect.process()