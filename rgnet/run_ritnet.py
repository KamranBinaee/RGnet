#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:37:59 2021

@author: Kamran Binaee
"""

import PIL
from PIL import Image
import torch
# from .dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from externals.dataset import transform
import os
from externals.opt import parse_args
from externals.models import model_dict
from tqdm import tqdm
from externals.utils import get_predictions
import cv2
import imageio
import pickle

#%%

def CannyThreshold(src, val1, val2):
    max_lowThreshold = 100
    window_name = 'Edge Map'
    title_trackbar = 'Min Threshold:'
    ratio = 3
    kernel_size = 3
    low_threshold = val1
    high_threshold = val2
    img_blur = cv2.blur(src, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, high_threshold, kernel_size) # low_threshold*ratio
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))

    # cv.imshow(window_name, dst)
    return detected_edges.astype(src.dtype)#dst

def contours_to_ellipse(contours):

    return ellipse_dict

# def image_to_contours(labels):
#     # parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
#     # parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
#     # args = parser.parse_args()
#     src = cv.imread(cv.samples.findFile(args.input))
#     if src is None:
#         print('Could not open or find the image: ', args.input)
#         exit(0)
#     return contours    


if __name__ == '__main__':

    args = parse_args()

    save_video = True
    save_labels = False
    save_directory = "/hdd01/kamran_sync/Projects/Deep_Pupil_Tracking/Results"
    gamma = args.gamma
    video_file = args.video_file
    session_id = video_file[-28:-9]
    # RIT-0: Default Pre-trained RITNet model
    method = "RIT-0"
    # Prepare the empty lists for storing pupil positions and the timestamps
    pupil_x = []
    pupil_y = []
    pupil_size = []
    pupil_index = []

    # Instantiate the video capture from opencv to read the eye video, total number of frames, frame width and height
    cap = cv2.VideoCapture(video_file)

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( 'Total Number of Frames: ', number_of_frames )
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('Frame Size :[', frame_height,frame_width, ']')
    start_index = 60*120
    end_index = 4*60*120 # number_of_frames
    print('Start/End Index :[', start_index, end_index, ']')


    # Instantiate the video capture from imageio in order to read the eye video
    vid = imageio.get_reader(video_file,  'ffmpeg')

    size = (frame_width, frame_height)

    eye_id = 0
    if "eye1" in video_file:
        eye_id = 1

    if (save_video == True):
        out_video_file = ("{}/{}_{}_{}_{}.mp4").format(save_directory, session_id,method,gamma, eye_id)
        print('output video file: {}'.format(out_video_file) )
        fourcc = 'mp4v'
        out = cv2.VideoWriter(out_video_file,cv2.VideoWriter_fourcc(*fourcc), 120, (1200,400)) #size
    else:
        print("\n No video output being saved!")

   
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    if args.useGPU:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    model = model_dict[args.model]
    model  = model.to(device)
    filename = args.load
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    counter=0
    
    # os.makedirs('vedb_test/labels/',exist_ok=True)
    # os.makedirs('vedb_test/output/',exist_ok=True)
    # os.makedirs('vedb_test/mask/',exist_ok=True)
    labels = []

    # out_video_file = ("{}/{}_{}_{}_{}_test.mp4").format(save_directory, session_id,method,gamma, eye_id)
    # print('output video file: {}'.format(out_video_file))
    # fourcc = 'mp4v'
    # out = cv2.VideoWriter(out_video_file,cv2.VideoWriter_fourcc(*fourcc), 30, (1200,400))

    with torch.no_grad():
        #for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
        for i in range(start_index, end_index,4):
    
            # Read the next frame from the video.
            raw_image = vid.get_data(i)
            # Switch the color channels since opencv reads the frames under BGR and the imageio uses RGB format
            raw_image[:, :, [0, 2]] = raw_image[:, :, [2, 0]]

            # Convert the image into grayscale
            gray = cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)

            pilimg =  Image.fromarray(raw_image).convert("L")
            if "eye0" in video_file:
                pilimg = pilimg.transpose(PIL.Image.FLIP_TOP_BOTTOM)


            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            #imagepath="model/test_image.png"
            # imagepath = '/hdd01/Deep_Gaze_Tracking/RIT-Net/vedb_data4/' + str(i) + '.jpg'
            #pilimg = Image.open(imagepath).convert("L")     #use L instead of Grayscale image
            H, W = pilimg.width , pilimg.height
           
            table = 255.0*(np.linspace(0, 1, 256)**gamma)  ##use Gamma correction

            # Todo: Pass a flag for gamma if 0 or 1 ignore
            pilimg = cv2.LUT(np.array(pilimg), table)
            img = clahe.apply(np.array(np.uint8(pilimg)))    ##use CLAHE
            # img = np.array(np.uint8(pilimg))
            #plt.imsave('/hdd01/Deep_Gaze_Tracking/RIT-Net/vedb_data4/CLAHE_{}.png'.format(index[i]),img)
            #print(img)
            img = Image.fromarray(img)      
            img = transform(img).unsqueeze(1).to(device)      ##Transform and add dimension so final dimension is 1,1,192,192
            predict = get_predictions(model(img))       ##convert into labels

            #print('predict:', np.unique(predict), end="\r", flush=True)
            print("Progress {0:.2f}% predict: {s}".format(i*100/end_index, s = str(np.unique(predict))), end="\r", flush=True)

            j = 0    
            pred_img = predict[j].cpu().numpy()/3.0
            if (save_labels == True):
                labels.append(pred_img)
            #print(pred_img)
            inp = img[j].squeeze() * 0.5 + 0.5
            #print('inp:', inp.shape)
            img_orig = np.clip(inp.cpu(),0,1)
            img_orig = np.array(img_orig)
            #combine = np.hstack([img_orig,pred_img])
            # plt.imsave('/hdd01/Deep_Gaze_Tracking/RIT-Net/output4/{}.png'.format(index[i]),pred_img)
            numpy_horizontal = np.hstack((gray/255, img_orig))
            numpy_horizontal = np.hstack((numpy_horizontal, pred_img))
            title = ("Input / Contrast-Enhanced gamma={} / Output").format(gamma)
            cv2.imshow(title, numpy_horizontal)#cv2.hconcat([gray.astype(np.uint8), pred_img])
            #cv2.imwrite(filename, raw_image)
            # print('\nmax/min ', pred_img.max(), pred_img.min())
            # print(np.unique(pred_img))
            # # print('shape', type(pred_img))
            # np.array(pred_img, dtype =uint8)
            model_output = cv2.cvtColor((pred_img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            #print(np.unique(model_output))
            # thresholds = [0,  85, 170, 255]
            thresholds = [10,  20, 60, 80]
            edge_0 = CannyThreshold(model_output, thresholds[0], thresholds[1])
            edge_1 = np.hstack((CannyThreshold(model_output, thresholds[1], thresholds[2]), edge_0))
            edge_2 = np.hstack((CannyThreshold(model_output, thresholds[2], thresholds[3]), edge_1))
            edge_3 = np.hstack((CannyThreshold(model_output, thresholds[3], 256), edge_2))
            cv2.imshow("contour", edge_3)
            input_image = cv2.cvtColor((img_orig*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            #labels = np.append([[labels]],pred_img, axis=0)
            if (save_labels == True):
                print("shape: ", np.asarray(labels).shape)

            #numpy_horizontal = np.hstack((gray, pred_img))
            #print(cv2.hconcat([gray.astype(np.uint8), pred_img]).shape)
            #final_frame = (numpy_horizontal*255).astype(np.uint8)

            #img = cv2.resize(img,(400,400))
            #final_frame = np.concatenate((gray, pred_img), axis=1)
            #print("max/min {} {} {}".format(gray.max(), gray.min(), gray.shape))
            # print("max/min {} {} {}".format(pred_img.max(), pred_img.min(), pred_img.shape))
            a = np.concatenate((raw_image, input_image), axis=1)
            final_frame = np.concatenate((a, model_output), axis=1)
            # print("Final: ", final_frame.shape)
            if (save_video == True):
                out.write(final_frame) # cv2.merge([pred_img, pred_img,pred_img])

            #out.write(final_frame)#cv2.hconcat([gray, pred_img]) 
            # out.write(raw_image)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                # out.release()
                print("\nQ detected!!")
                break

print('\n Inference Done!', i)
print("Final Shape", np.asarray(labels).shape)
# Close all the opencv image frame windows opened
cv2.destroyAllWindows()
cap.release()
vid.close()
if (save_video == True):
    out.release()
# out_pickle_file = ("{}/{}_{}_{}_{}.pickle").format(save_directory, session_id,method,gamma, eye_id)
# print('output video file: {}'.format(out_pickle_file) )

# f = open(out_pickle_file, 'wb')
# pickle.dump(np.asarray(labels, dtype=np.uint8), f, pickle.HIGHEST_PROTOCOL)
# f.close()
 
    #os.rename('test',args.save)

# How to run the code through terminal
# python3 test_vedb3.py --video_file /hdd01/kamran_sync/vedbcloud0/staging/2021_02_25_16_28_15/eye1.mp4 --gamma 0.5 --model densenet --load best_model.pkl --bs 4


# import cv2
# from PIL import Image
# video_file = "/home/kamran/temp_sync/vedbcloud0/staging/2021_05_11_16_58_35/eye1.mp4"
# # Instantiate the video capture from opencv to read the eye video, total number of frames, frame width and height
# cap = cv2.VideoCapture(video_file)

# number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print( 'Total Number of Frames: ', number_of_frames )
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print('Frame Size :[', frame_height,frame_width, ']')
# start_index = 60*120
# end_index = 2*60*120 # number_of_frames
# print('Start/End Index :[', start_index, end_index, ']')



# clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
# #imagepath="model/test_image.png"
# # imagepath = '/hdd01/Deep_Gaze_Tracking/RIT-Net/vedb_data4/' + str(i) + '.jpg'
# pilimg = Image.open(imagepath).convert("L")     #use L instead of Grayscale image
# H, W = pilimg.width , pilimg.height

# table = 255.0*(np.linspace(0, 1, 256)**gamma)  ##use Gamma correction
# pilimg = cv2.LUT(np.array(pilimg), table)

# img = clahe.apply(np.array(np.uint8(pilimg)))    ##use CLAHE
# #plt.imsave('/hdd01/Deep_Gaze_Tracking/RIT-Net/vedb_data4/CLAHE_{}.png'.format(index[i]),img)
# #print(img)
# img = Image.fromarray(img)      
# img = transform(img).unsqueeze(1).to(device)      ##Transform and add dimension so final dimension is 1,1,192,192
# predict = get_predictions(model(img))       ##convert into labels

# #print('predict:', np.unique(predict), end="\r", flush=True)
# print("Progress {0:.2f}% predict: {s}".format(i*100/end_index, s = str(np.unique(predict))), end="\r", flush=True)

# j = 0    
# pred_img = predict[j].cpu().numpy()/3.0
# labels.append(pred_img)
# #print(pred_img)
# inp = img[j].squeeze() * 0.5 + 0.5
# #print('inp:', inp.shape)
# img_orig = np.clip(inp.cpu(),0,1)
# img_orig = np.array(img_orig)
# #combine = np.hstack([img_orig,pred_img])
# # plt.imsave('/hdd01/Deep_Gaze_Tracking/RIT-Net/output4/{}.png'.format(index[i]),pred_img)
# numpy_horizontal = np.hstack((gray/255, img_orig))
# numpy_horizontal = np.hstack((numpy_horizontal, pred_img))
# title = ("Input / Contrast-Enhanced gamma={} / Output").format(gamma)
# cv2.imshow(title, numpy_horizontal)#cv2.hconcat([gray.astype(np.uint8), pred_img])
# #cv2.imwrite(filename, raw_image)
# # print('\nmax/min ', pred_img.max(), pred_img.min())
# # print(np.unique(pred_img))
# # # print('shape', type(pred_img))
# # np.array(pred_img, dtype =uint8)
# model_output = cv2.cvtColor((pred_img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
# input_image = cv2.cvtColor((img_orig*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
# #labels = np.append([[labels]],pred_img, axis=0)
# print("shape: ", np.asarray(labels).shape)

# if (save_video == True):
#     out.write(cv2.merge([pred_img, pred_img,pred_img]))


## Aayuush's code

# clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
# imagepath="model/test_image.png"
# pilimg = Image.open(imagepath).convert("L")     #use L instead of Grayscale image
# H, W = pilimg.width , pilimg.height

# table = 255.0*(np.linspace(0, 1, 256)**0.8)  ##use Gamma correction
# pilimg = cv2.LUT(np.array(pilimg), table)
       
# img = clahe.apply(np.array(np.uint8(pilimg)))    ##use CLAHE
# img = Image.fromarray(img)      
# img = transform(img).unsqueeze(1).to(device)      ##Transform and add dimension so final dimension is 1,1,192,192
# prediction = get_predictions(model(img)))       ##convert into labels