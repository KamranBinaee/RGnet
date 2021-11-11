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
from ellipse import LsqEllipse
import copy
from skimage.measure import EllipseModel, ransac
import pandas as pd


#%%

def fit_ellipse(raw_frame, input_image, ellipse_model, frame_number, pupil_df, eye_id, ts):
    residual = None
    detection_method = "retrained_ritnet"
    model = ellipse_model
    # raw_frame = np.load(array_path + "/" + array_files[i])
    # print(raw_frame.shape)
    # a = np.array([raw_frame, raw_frame, raw_frame])
    # raw_frame = a
    # print(raw_frame.shape)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    raw_image = copy.deepcopy(raw_frame)

    pupil_mask = np.zeros((frame_width, frame_height), dtype=np.uint8)
    iris_mask = np.zeros((frame_width, frame_height), dtype=np.uint8)
    sclera_mask = np.zeros((frame_width, frame_height), dtype=np.uint8)
    skin_mask = np.zeros((frame_width, frame_height), dtype=np.uint8)

    # Read the next frame from the video.
    pupil_mask[raw_image == 3] = 255
    iris_mask[raw_image == 2] = 255
    sclera_mask[raw_image == 1] = 255
    skin_mask[raw_image == 0] = 80

    th1 = 10
    th2 = 220
    edges_pupil = cv2.Canny(pupil_mask, th1, th2)
    edges_iris = cv2.Canny(iris_mask, th1, th2)
    edges_sclera = cv2.Canny(sclera_mask, th1, th2)
    edges_skin = cv2.Canny(skin_mask, th1, th2)

    # Convert the image into grayscale
    pupil_center = np.zeros((frame_width, frame_height, 3))
    #     pupil_center[np.where(raw_image[:,:,1] == 255)] =
    #     x,y,z = np.where(raw_image == [0,255,0])
    r_query, g_query, b_query = (30, 150, 30)
    p_value = 255
    i_value = 170
    s_value = 85
    k_value = 0
    x, y = np.where(edges_iris == 255)  # (raw_image == i_value)
    if len(x) > 5:
        pupil_center[x, y, :] = np.array([255, 0, 255])
        input_image[x, y, :] = np.array([255, 0, 255])

        pupil_centeroid = np.array([np.mean(x), np.mean(y)])

        X = np.array(list(zip(y, x)))

        model.estimate(X)
        center_x, center_y, width, height, phi = np.round(model.params)
        # xc, yc, a, b, theta

        # To extract the likelihood I draw a blue ellipse to act as a mask
        center_coordinates = (int(center_x), int(center_y))
        axesLength = (int(width), int(height))
        angle = np.rad2deg(phi)
        startAngle = 0
        endAngle = 360
        # Blue color in BGR
        color = (0, 0, 255)
        # Line thickness of -1 px
        thickness = 2  # -1
        # Using cv2.ellipse() method
        # Draw a ellipse with blue line borders of thickness of -1 px
        pupil_center = cv2.ellipse(pupil_center, center_coordinates, axesLength, angle, startAngle, endAngle, color,
                                   thickness)
    #         input_image = cv2.ellipse(input_image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)

    x, y = np.where(edges_pupil == 255)  # (raw_image == p_value)
    if len(x) > 5:
        pupil_center[x, y, :] = np.array([255, 255, 0])
        #         input_image[x,y,:] = np.array([255, 255,0])

        X = np.array(list(zip(y, x)))

        model.estimate(X)
        center_x, center_y, width, height, phi = np.round(model.params)
        # xc, yc, a, b, theta

        # centroid_list.append((center_x, center_y))
        # width_list.append(width)
        # height_list.append(height)
        # angle_list.append(np.rad2deg(phi))
        # confidence_list.append(np.mean(model.residuals(X)))
        # index_list.append(i)

        if (len(X) > 20):
            model.estimate(X)
            residual = np.mean(model.residuals(X))
            center_x, center_y, width, height, phi = np.round(model.params)
            final_model = model
            if residual > 2:
                ransac_model, inliers = ransac(X, EllipseModel, 20, 3, max_trials=30)
                center_x, center_y, width, height, phi = np.round(ransac_model.params)
                # xc, yc, a, b, theta
                residual = np.mean(ransac_model.residuals(X))
                final_model = model
            # centroid_list_r.append((center_x, center_y))
            # width_list_r.append(width)
            # height_list_r.append(height)
            # angle_list_r.append(np.rad2deg(phi))
            # confidence_list_r.append(residual)
            # index_list_r.append(i)

            center_coordinates = (int(center_x), int(center_y))
            axesLength = (int(width), int(height))
            angle = np.rad2deg(phi)
            startAngle = 0
            endAngle = 360
            # Blue color in BGR
            color = (0, 255, 255)
            # Line thickness of -1 px
            thickness = 1  # -1
            # Using cv2.ellipse() method
            # Draw a ellipse with blue line borders of thickness of -1 px
            pupil_center = cv2.ellipse(pupil_center, center_coordinates, axesLength, angle, startAngle, endAngle, color,
                                       thickness=2)
            input_image = cv2.ellipse(input_image, center_coordinates, axesLength, angle, startAngle, endAngle, color,
                                      thickness=4)
            center_x, center_y, width, height, phi = final_model.params
            pupil_df["pupil_timestamp"].loc[frame_number] = ts
            pupil_df["eye_index"].loc[frame_number] = int(frame_number)
            pupil_df["eye_id"].loc[frame_number] = eye_id
            pupil_df["confidence"].loc[frame_number] = residual
            pupil_df["norm_pos_x"].loc[frame_number] = center_x / 400.0
            pupil_df["norm_pos_y"].loc[frame_number] = center_y / 400.0
            pupil_df["diameter"].loc[frame_number] = np.power(np.power(width,2) + np.power(height,2),0.5)
            pupil_df["method"].loc[frame_number] = detection_method
            pupil_df["ellipse_center_x"].loc[frame_number] = center_x
            pupil_df["ellipse_center_y"].loc[frame_number] = center_y
            pupil_df["ellipse_axis_a"].loc[frame_number] = width
            pupil_df["ellipse_axis_b"].loc[frame_number] = height
            pupil_df["ellipse_angle"].loc[frame_number] = angle
            # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (250, 20)
    # fontScale
    fontScale = 0.5
    # Blue color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    if residual is None:
        residual = 1000
    pupil_center = cv2.putText(pupil_center, "Residual: " + str(np.round(residual, 2)), org, font, fontScale, color,
                               thickness, cv2.LINE_AA)
    final_frame = np.concatenate((input_image.astype(np.uint8), pupil_center.astype(np.uint8)), axis=1)
    return final_frame, pupil_df

def creat_output_data(df_index):
    df_keys = ["pupil_timestamp", "eye_index", "world_index", "eye_id", "confidence", "norm_pos_x",
                       "norm_pos_y",
                       "diameter", "method", "ellipse_center_x", "ellipse_center_y", "ellipse_axis_a", "ellipse_axis_b",
                       "ellipse_angle"] #,
                    # "diameter_3d", "model_confidence", "model_id", "sphere_center_x",
                    #    "sphere_center_y", "sphere_center_z", "sphere_radius", "circle_3d_center_x",
                    #    "circle_3d_center_y", "circle_3d_center_z", "circle_3d_normal_x", "circle_3d_normal_y",
                    #    "circle_3d_normal_z", "circle_3d_radius", "theta", "phi", "projected_sphere_center_x",
                    #    "projected_sphere_center_y", "projected_sphere_axis_a", "projected_sphere_axis_b",
                    #    "projected_sphere_angle"]
    df = pd.DataFrame(columns=df_keys, index=df_index, dtype=np.float64)
    return df

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
    root_directory = "/hdd01/kamran_sync/staging/"
    all_sessions =[#"2021_07_05_15_25_46", # #UNR, indoor watching TV, Allie
                   # "2021_07_05_13_09_00", #UNR, indoor calibration, Allie
                   # "2021_05_15_08_44_13", #UNR, indoor fusbal, Pogen
                   # "2021_07_05_12_43_05", #UNR, making coffee, Allie
                   # "2021_07_05_11_40_12", #UNR, house clean up, Allie
                   # "2021_06_24_10_28_38", #UNR, working hands, Matt
                   # "2021_05_23_14_33_31", #UNR, indoor board game, Mark
                   # "2021_05_15_11_27_29", #UNR, indoor game, Pogen
                   # "2021_04_06_18_37_22", #UNR, indoor office, Kamran
                   # "2021_02_04_16_42_40", #UNR, indoor walking, Kaylie
                   # "2021_10_21_15_51_33", #UNR, indoor Lab meeting, Mark

                    # "2021_04_29_11_54_15", #Bates, work on laptop, Juliet
                    # "2021_04_14_15_05_21", #Bates, indoor water plants, Jennifer
                    # "2021_03_31_12_58_47", #Bates, indoor cooking, Jennifer
                    # "2021_03_24_09_37_30", #Bates, indoor shopping, Michelle
                    # "2021_04_14_15_51_11", #Bates, Shopping, Jennifer
                    # "2021_10_30_16_39_08",	#Bates, Playing Guitar, Michelle


                    # "2021_06_24_12_57_27", #UNR, skate boarding outdoor, Matt
                    # "2021_06_24_12_27_16", #UNR, walking outdoor, Matt
                    # "2021_06_07_13_31_48", #UNR, outdoor calibration, Kaylie
                    # "2021_05_11_16_58_35", #UNR, outdoor walking, Pogen
                    # "2021_05_28_08_54_15", #UNR, outdoor walking, Mark

                    # "2021_11_04_14_12_36", #Bates, Swing, Jennifer
                    # "2021_10_19_16_07_29", #Bates, outdoor hiking, Jenn
                    # "2021_10_24_13_35_37", #Bates, outdoor car ride, Michelle
                    # "2021_10_24_13_46_50", #Bates, outdoor walking, Jenn
                    # "2021_05_22_14_02_34", #Bates, outdoor walking, Michelle
                    # "2021_04_28_14_24_57", #Bates, Driving walking, Juliet
                    # "2021_04_14_17_23_20", #Bates, outdoor Skate boarding, Juliet
                    ]
    for this_session in all_sessions:
        for eye_id in [0, 1]:

            video_file = root_directory + this_session + "/eye" + str(eye_id) + ".mp4"
            print("Running pupil detection for:", video_file)

            save_video = True
            save_labels = False
            save_directory = "/hdd01/kamran_sync/Projects/Deep_Pupil_Tracking/Results3"
            gamma = args.gamma
            # video_file = args.video_file
            session_id = video_file[-28:-9]
            # RIT-0: Default Pre-trained RITNet model
            method = "RIT-retrained"
            # Prepare the empty lists for storing pupil positions and the timestamps
            pupil_x = []
            pupil_y = []
            pupil_size = []
            pupil_index = []

            # Instantiate the video capture from opencv to read the eye video, total number of frames, frame width and height
            cap = cv2.VideoCapture(video_file)

            number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print('Total Number of Frames: ', number_of_frames)
            pupil_df = creat_output_data(np.arange(number_of_frames))
            time_stamp = np.load(video_file.replace(".mp4", "_timestamps.npy"))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print('Frame Size :[', frame_height,frame_width, ']')
            start_index = 0 #1*60*120
            end_index = min(5*60*120, number_of_frames)
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
                # out = cv2.VideoWriter(out_video_file,cv2.VideoWriter_fourcc(*fourcc), 120, (1200,400)) #size
                out = cv2.VideoWriter(out_video_file,cv2.VideoWriter_fourcc(*fourcc), 30, (800,400)) #size
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

            ellipse_model = EllipseModel()
            with torch.no_grad():
                #for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
                for i in range(start_index, end_index, 4):

                    # Read the next frame from the video.
                    raw_image = vid.get_data(i)
                    # Switch the color channels since opencv reads the frames under BGR and the imageio uses RGB format
                    raw_image[:, :, [0, 2]] = raw_image[:, :, [2, 0]]

                    # Convert the image into grayscale
                    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

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
                    img = raw_image
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
                    combined_image, pupil_df = fit_ellipse(predict[j].cpu().numpy(), gray, ellipse_model, i, pupil_df, eye_id, time_stamp[i])
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
                    # thresholds = [10,  20, 60, 80]
                    # edge_0 = CannyThreshold(model_output, thresholds[0], thresholds[1])
                    # edge_1 = np.hstack((CannyThreshold(model_output, thresholds[1], thresholds[2]), edge_0))
                    # edge_2 = np.hstack((CannyThreshold(model_output, thresholds[2], thresholds[3]), edge_1))
                    # edge_3 = np.hstack((CannyThreshold(model_output, thresholds[3], 256), edge_2))
                    # # cv2.imshow("contour", edge_3)

                    cv2.imshow("Model Result", combined_image)
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
                        out.write(combined_image) # final_frame # cv2.merge([pred_img, pred_img,pred_img])

                    #out.write(final_frame)#cv2.hconcat([gray, pred_img])
                    # out.write(raw_image)
                    if cv2.waitKey(2) & 0xFF == ord('q'):
                        # out.release()
                        print("\nQ detected!!")
                        break

            print('\n Inference Done!', i)
            # print("Final Shape", np.asarray(labels).shape)
            # Close all the opencv image frame windows opened
            cv2.destroyAllWindows()
            cap.release()
            vid.close()
            if (save_video == True):
                out.release()
            print('\n Video File Saved!')
            pupil_df.to_pickle(out_video_file.replace(".mp4", ".pkl"))
            print("\nPickle File Saved!!!")
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