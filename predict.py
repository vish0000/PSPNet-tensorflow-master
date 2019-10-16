import tensorflow as tf
import cv2
import imutils
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

from model import PSPNet101, PSPNet50
from tools import *

tf.reset_default_graph()


import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='once')

# Input and output files
input_directory = "input";
output_directory = "output";

# Indoor model
#ADE20k_param = {'crop_size': [473, 473],
#                'num_classes': 150, 
#                'model': PSPNet50,
#                'weights_path': './model/pspnet50-ade20k/model.ckpt-0'}
# Outdoor model
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101,
                    'weights_path': './model/pspnet101-cityscapes/model.ckpt-0'}

IMAGE_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

'''
# Run through all images
for item in os.listdir(input_directory):
    # Create image path
    image_path = './' + input_directory + '/' + item
    im = Image.open(image_path)
    
    vishal = im.getdata()
    
    #print(vishal[0])
    if not item.startswith('.') and os.path.isfile(image_path):                

        # NOTE: If you want to inference on indoor data, change this value to `ADE20k_param`
        param = cityscapes_param 
        img_np, filename = load_img(image_path)
        img_shape = tf.shape(img_np)
        h, w = (tf.maximum(param['crop_size'][0], img_shape[0]), tf.maximum(param['crop_size'][1], img_shape[1]))
        img = preprocess(img_np, h, w)
        # Create network.
        PSPNet = param['model']
        net = PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])
        raw_output = net.layers['conv6']
        # Predictions.
        raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
        raw_output_up = tf.argmax(raw_output_up, dimension=3)
        pred = decode_labels(raw_output_up, img_shape, param['num_classes'])
        # Init tf Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        ckpt_path = param['weights_path']
        loader = tf.train.Saver(var_list=tf.global_variables())
        loader.restore(sess, ckpt_path)
        print("Restored model parameters from {}".format(ckpt_path))
        # Run and get result image
        preds = sess.run(pred)
        # Write output image
        cv2.imwrite(output_directory + '/' + item, preds[0])
        # Restore for the next file
        tf.reset_default_graph() 
# '''   
        
image_path1 = './' + input_directory + '/test1.png'
im1 = Image.open(image_path1)
        
vishal = im1.getdata()
print (vishal[0])
print(len(vishal))
image_path2 = './' + output_directory + '/test1.png'
im2 = Image.open(image_path2) 
vishal2 = im2.getdata()
print (vishal2[0])

pixels = im2.load()

new_pixels = im1.load()


print("foeloop")

for i in range(im2.size[0]):    # for every col:
    for j in range(im2.size[1]):
#        #for car 
#        if(pixels[i,j][0] == 142 and  pixels[i,j][1] == 0 and  pixels[i,j][2] == 0 ):
#                new_pixels[i,j] = (255,255,255)
#         #for side_walk
#         if(pixels[i,j][0] == 231 and  pixels[i,j][1] == 35 and  pixels[i,j][2] == 244 ):
#                new_pixels[i,j] = (255,255,255)
        
#         #for bulding
#         if(pixels[i,j][0] == 69 and  pixels[i,j][1] == 69 and  pixels[i,j][2] == 69 ):
#                new_pixels[i,j] = (255,255,255)

#         #for road
#         if(pixels[i,j][0] == 128 and  pixels[i,j][1] == 64 and  pixels[i,j][2] == 128 ):
#                new_pixels[i,j] = (255,255,255)
        
         #for road
         if(pixels[i,j][0] == 153 and  pixels[i,j][1] == 153 and  pixels[i,j][2] == 153 ):
                new_pixels[i,j] = (255,255,255)
                
       



        
im1.save("./vishal/pole.png")      
   
       
       


