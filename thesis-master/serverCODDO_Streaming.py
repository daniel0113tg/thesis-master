##Libraries
import os
import glob
import math
import argparse
import pytictoc
import common
import imagezmq
import copy
import numpy as np
import pandas as pd
import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Imaget
from itertools import chain
from keras import preprocessing
from keras.applications.vgg16 import preprocess_inpu
from create_neural_net_model import create_cnn_model
from create_neural_net_model import create_neural_net_model
from imutils import build_montages
from datetime import datetime
from utils import CvFpsCalc


# initialize the ImageHub object
imageHub = imagezmq.ImageHub()
frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now

# stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD
frames = [] 
framesXCNN = []
L_hand_landmarks = []
R_hand_landmarks = []



# 0 maps to 'P' (speaking), 1 maps to 'S' (signing), 2 maps to 'n' (other) 
CLASS_MAP = ['P', 'S', 'n']


##	python3 generate_CNN_features.py --input=sld/frames --output=sld/frames_cnnfc1 --groundtruth=sld/groundtruth.txt  --fc1_layer=True

def generate_CNN_features(frame,time ,cnn_model):
    tt = pytictoc.TicToc()
    tt.tic()
    # load the image and convert to numpy 3D array
    img_pil = preprocessing.image.array_to_img(frame)
    img = np.array(preprocessing.image.array_to_img(img_pil))

    # Note that we don't scale the pixel values because VGG16 was not trained with normalised pixel values!
    # Instead we use the pre-processing function that comes specifically with the VGG16
    X = preprocess_input(img)

    X = np.expand_dims(X, axis=0)       
    # package as a batch of size 1, by adding an extra dimension
    # generate the CNN features for this batch
    print(".", end='', flush=True)
    X_cnn = cnn_model.predict_on_batch(X)
    tt.toc()
    print('\n\nReady')
    return [time, X_cnn]
                

def visualise_labels(gt_list, title_str=""):
    fig = plt.figure()
    currentAxis = plt.gca()
    for (i, gt) in enumerate(gt_list):
        col = 'gray'
        if gt == 'P':
            col = 'green'
        if gt == 'S':
            col = 'red'
        if gt == 'n':
            col = 'yellow'
        currentAxis.add_patch(Rectangle((i, 0), 1, 4, alpha=1, fill=True, color=col))
    plt.xlim(0, len(gt_list))
    plt.ylim(0, 4)
    plt.title(title_str)
    plt.tight_layout()
    plt.show()
    return fig


def detect_sign(frame, timesteps, image_data_shape, video_data_shape, rnn_input_shape, include_cnn_fc1_layer, model_weights_file, output_path):

    # load the top RNN part of the model, without the convolutional base
    model = create_neural_net_model(image_data_shape, video_data_shape, rnn_input_shape,
            include_convolutional_base=False, include_cnn_fc1_layer=include_cnn_fc1_layer, include_top_layers=True, rnn_model_weights_file=model_weights_file)

    # select the groundtruth rows for this video
    print('Processing frame {} ...'.format(frame[0]))

    cnn_files = []
    gt_labels = []
    pred_labels = []
    frame_numbers = []
    frame_numbers.append(frame[0])
    cnn_files.append(frame[1])
    gt_labels.append(('?'))

    print('\n\n')
    assert len(cnn_files) == len(gt_labels) == len(frame_numbers), 'logical error!!'

    pred_prob = np.zeros(len(frame_numbers))          # probability of the maximal class
    pred_probs = np.zeros((len(frame_numbers), 3))    # probabilities of all classes

    # scan the video with a sliding window of T timesteps
    left_win_size = int(timesteps / 2)
    right_win_size = timesteps -1 - left_win_size
    for (i, fr) in enumerate(frame_numbers):
        window_ndx = chain(range(i - left_win_size, i), range(i, i + right_win_size + 1))
        window_ndx = list(window_ndx)
        window_ndx = [0 if i < 0 else i for i in window_ndx]        # take care of loop boundary conditions
        window_ndx = [len(frame_numbers) - 1 if i >= len(frame_numbers) else i for i in window_ndx]  # take care of loop boundary conditions

        # get the CNN features Cuesiton
        X = []
        for (j, ndx) in enumerate(window_ndx):
            print(cnn_files[ndx])
            dt = cnn_files[ndx]
   # has shape (timesteps, CNN feature vector length)
            X.append(dt[0, ...])
        X = np.array(X)
        # package the input data as a batch of size 1
        X = np.expand_dims(X, axis=0)       # a batch of 1, adding an extra dimension
        # process...
        answer = model.predict(X)
        
        # find the maximum of the predictions (& decode from one-hot-encoding for groundthruth labels)
        for batch_i in range(0, len(answer)):     # we have an answer for each batch (1 answer in this case)
            predicted_class = np.argmax(answer[batch_i])
            predicted_label = CLASS_MAP[predicted_class]
            pred_labels.append(predicted_label)
            pred_prob[i] = answer[batch_i, np.argmax(answer[batch_i])]   # probability for the predicted class
            pred_probs[i, 0:3] = answer                                  # probabilities for all the classes
        print(pred_labels[-1], end='', flush=True)
    print('\n\n')
    assert len(pred_labels) == len(gt_labels) == len(pred_prob) == len(pred_probs), 'logical error during prediction stage!!'
    print("LABELS\n")
    print(pred_labels)
    print("\n")
    exists_sign = False
    if output_path:
        results_file = open(os.path.join(output_path, video_id+'_signing.txt'), 'w')
        results_file.write('video_id,frame_number,predicted_label,\n')
        for k in range(len(frame_numbers)):
            if(pred_labels[k] == "S"):
                exists_sign = True
                results_file.write('%s,%d,%s\n' % (frame[0], frame[1], pred_labels[k]))
        results_file.close()

    print('\nready')
    return exists_sign



def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_hands_landmarks(time,
                        image,
                         cx,
                         cy,
                         landmarks,
                         upper_body_only,
                         handedness_str='R',):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    landmark_point.append(time)
    landmark_point.append(handedness_str)

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

        if not upper_body_only:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, landmark_point[4], landmark_point[5], (0, 255, 0), 2)
        cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)

        # 人差指
        cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)
        cv.line(image, landmark_point[8], landmark_point[9], (0, 255, 0), 2)
        cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)

        # 中指
        cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)
        cv.line(image, landmark_point[12], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)

        # 薬指
        cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)
        cv.line(image, landmark_point[16], landmark_point[17], (0, 255, 0), 2)
        cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)

        # 小指
        cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)
        cv.line(image, landmark_point[20], landmark_point[21], (0, 255, 0), 2)
        cv.line(image, landmark_point[21], landmark_point[22], (0, 255, 0), 2)

        # 手の平
        cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)
        cv.line(image, landmark_point[4], landmark_point[7], (0, 255, 0), 2)
        cv.line(image, landmark_point[7], landmark_point[11], (0, 255, 0), 2)
        cv.line(image, landmark_point[11], landmark_point[15], (0, 255, 0), 2)
        cv.line(image, landmark_point[15], landmark_point[19], (0, 255, 0), 2)
        cv.line(image, landmark_point[19], landmark_point[2], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        if handedness_str == 'L':
            L_hand_landmarks.append(landmark_point)
        if handedness_str == 'R':
            R_hand_landmarks.append(landmark_point)
        cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv.putText(image, handedness_str, (cx - 6, cy + 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
    
    return image


def draw_face_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        cv.circle(image, (landmark_x, landmark_y), 1, (0, 255, 0), 1)

    if len(landmark_point) > 0:

        # 左眉毛(55：内側、46：外側)
        cv.line(image, landmark_point[55], landmark_point[65], (0, 255, 0), 2)
        cv.line(image, landmark_point[65], landmark_point[52], (0, 255, 0), 2)
        cv.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
        cv.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2)

        # 右眉毛(285：内側、276：外側)
        cv.line(image, landmark_point[285], landmark_point[295], (0, 255, 0),
                2)
        cv.line(image, landmark_point[295], landmark_point[282], (0, 255, 0),
                2)
        cv.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),
                2)
        cv.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),
                2)

        # 左目 (133：目頭、246：目尻)
        cv.line(image, landmark_point[133], landmark_point[173], (0, 255, 0),
                2)
        cv.line(image, landmark_point[173], landmark_point[157], (0, 255, 0),
                2)
        cv.line(image, landmark_point[157], landmark_point[158], (0, 255, 0),
                2)
        cv.line(image, landmark_point[158], landmark_point[159], (0, 255, 0),
                2)
        cv.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),
                2)
        cv.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),
                2)
        cv.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),
                2)

        cv.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),
                2)
        cv.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),
                2)
        cv.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),
                2)
        cv.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),
                2)
        cv.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),
                2)
        cv.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),
                2)
        cv.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),
                2)

        # 右目 (362：目頭、466：目尻)
        cv.line(image, landmark_point[362], landmark_point[398], (0, 255, 0),
                2)
        cv.line(image, landmark_point[398], landmark_point[384], (0, 255, 0),
                2)
        cv.line(image, landmark_point[384], landmark_point[385], (0, 255, 0),
                2)
        cv.line(image, landmark_point[385], landmark_point[386], (0, 255, 0),
                2)
        cv.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),
                2)
        cv.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),
                2)
        cv.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),
                2)

        cv.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),
                2)
        cv.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),
                2)
        cv.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),
                2)
        cv.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),
                2)
        cv.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),
                2)
        cv.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),
                2)
        cv.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),
                2)

        # 口 (308：右端、78：左端)
        cv.line(image, landmark_point[308], landmark_point[415], (0, 255, 0),
                2)
        cv.line(image, landmark_point[415], landmark_point[310], (0, 255, 0),
                2)
        cv.line(image, landmark_point[310], landmark_point[311], (0, 255, 0),
                2)
        cv.line(image, landmark_point[311], landmark_point[312], (0, 255, 0),
                2)
        cv.line(image, landmark_point[312], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
        cv.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
        cv.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
        cv.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
        cv.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

        cv.line(image, landmark_point[78], landmark_point[95], (0, 255, 0), 2)
        cv.line(image, landmark_point[95], landmark_point[88], (0, 255, 0), 2)
        cv.line(image, landmark_point[88], landmark_point[178], (0, 255, 0), 2)
        cv.line(image, landmark_point[178], landmark_point[87], (0, 255, 0), 2)
        cv.line(image, landmark_point[87], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
        cv.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),
                2)
        cv.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),
                2)
        cv.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),
                2)
        cv.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),
                2)

    return image


def draw_pose_landmarks(image, landmarks, upper_body_only, visibility_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 右目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 右肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 右手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 右手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 腰(左側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 右ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 左ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 右足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 左足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 右かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        if not upper_body_only:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)

    if len(landmark_point) > 0:
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][
                0] > visibility_th:
            cv.line(image, landmark_point[1][1], landmark_point[2][1],
                    (0, 255, 0), 2)
        if landmark_point[2][0] > visibility_th and landmark_point[3][
                0] > visibility_th:
            cv.line(image, landmark_point[2][1], landmark_point[3][1],
                    (0, 255, 0), 2)

        # 左目
        if landmark_point[4][0] > visibility_th and landmark_point[5][
                0] > visibility_th:
            cv.line(image, landmark_point[4][1], landmark_point[5][1],
                    (0, 255, 0), 2)
        if landmark_point[5][0] > visibility_th and landmark_point[6][
                0] > visibility_th:
            cv.line(image, landmark_point[5][1], landmark_point[6][1],
                    (0, 255, 0), 2)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][
                0] > visibility_th:
            cv.line(image, landmark_point[9][1], landmark_point[10][1],
                    (0, 255, 0), 2)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1],
                    (0, 255, 0), 2)

        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[13][1],
                    (0, 255, 0), 2)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 0), 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][
                0] > visibility_th:
            cv.line(image, landmark_point[15][1], landmark_point[17][1],
                    (0, 255, 0), 2)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
                0] > visibility_th:
            cv.line(image, landmark_point[17][1], landmark_point[19][1],
                    (0, 255, 0), 2)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
                0] > visibility_th:
            cv.line(image, landmark_point[19][1], landmark_point[21][1],
                    (0, 255, 0), 2)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[21][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
                0] > visibility_th:
            cv.line(image, landmark_point[16][1], landmark_point[18][1],
                    (0, 255, 0), 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
                0] > visibility_th:
            cv.line(image, landmark_point[18][1], landmark_point[20][1],
                    (0, 255, 0), 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
                0] > visibility_th:
            cv.line(image, landmark_point[20][1], landmark_point[22][1],
                    (0, 255, 0), 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[22][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[23][1],
                    (0, 255, 0), 2)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[24][1],
                    (0, 255, 0), 2)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[24][1],
                    (0, 255, 0), 2)

        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                    0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[25][1],
                        (0, 255, 0), 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                    0] > visibility_th:
                cv.line(image, landmark_point[25][1], landmark_point[27][1],
                        (0, 255, 0), 2)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                    0] > visibility_th:
                cv.line(image, landmark_point[27][1], landmark_point[29][1],
                        (0, 255, 0), 2)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                    0] > visibility_th:
                cv.line(image, landmark_point[29][1], landmark_point[31][1],
                        (0, 255, 0), 2)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                    0] > visibility_th:
                cv.line(image, landmark_point[24][1], landmark_point[26][1],
                        (0, 255, 0), 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                    0] > visibility_th:
                cv.line(image, landmark_point[26][1], landmark_point[28][1],
                        (0, 255, 0), 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                    0] > visibility_th:
                cv.line(image, landmark_point[28][1], landmark_point[30][1],
                        (0, 255, 0), 2)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                    0] > visibility_th:
                cv.line(image, landmark_point[30][1], landmark_point[32][1],
                        (0, 255, 0), 2)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)
    return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper_body_only', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument("--output", help="Output path where the results and figures will be saved to", default="")
    parser.add_argument("--model", help="File path and filename for the trained RNN model weights. File name should be *.h5", default="")

    args = parser.parse_args()

    return args
    
def main():
    
    args = get_args()
    imwidth = 224
    imheight = 224
    timesteps = 20
    fc1_layer = True
    image_data_shape = (imwidth, imheight, 3)   # width, height, channels
    video_clip_data_shape = (timesteps, imwidth, imheight, 3) 
    rnn_input_shape = (timesteps, 4096) if fc1_layer else (timesteps, 7, 7, 512)    # timesteps, CNN features width, CNN features height, CNN features channels
    model = create_cnn_model(image_data_shape, True)

    lastActive = {}
    lastActiveCheck = datetime.now()

    # 引数解析 #################################################################

    upper_body_only = args.upper_body_only
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

	    # モデルロード (Carga del modelo) #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        upper_body_only=upper_body_only,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    time = 0
    while True:
        display_fps = cvFpsCalc.get()
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')
        print("OK")
		# if a device is not in the last active dictionary then it means
		# that its a newly connected device
        if rpiName not in lastActive.keys():
        	print("[INFO] receiving data from {}...".format(rpiName))
		# record the last active time for the device from which we just
		# received a frame
        lastActive[rpiName] = datetime.now()

        # 検出実施 #########################s####################################
        print("Start ")
        image = frame
        XCNN = generate_CNN_features(image, time, model)
        sign = detect_sign( XCNN , timesteps, image_data_shape, video_clip_data_shape, rnn_input_shape, fc1_layer,model_weights_file=args.model, output_path=args.output)
        print(sign)
        if sign:
            debug_image = copy.deepcopy(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False  
            results = holistic.process(image)
            image.flags.writeable = True
            
            # Face Mesh ###########################################################
            face_landmarks = results.face_landmarks
            if face_landmarks is not None:
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, face_landmarks)
            # 描画
                debug_image = draw_face_landmarks(debug_image, face_landmarks)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            # Pose ###############################################################
            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, pose_landmarks)
                # 描画
                debug_image = draw_pose_landmarks(debug_image, pose_landmarks,upper_body_only)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            # Hands ###############################################################
            left_hand_landmarks = results.left_hand_landmarks
            right_hand_landmarks = results.right_hand_landmarks

            #左手
            if left_hand_landmarks is not None:
                # 手の平重心計算 (Calcular centor de gravedad de la palma izquierda)
                cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
                # Calcular el rectangulo delimitador
                brect = calc_bounding_rect(debug_image, left_hand_landmarks)
                #Dibujar
                debug_image = draw_hands_landmarks(time,debug_image, cx, cy,left_hand_landmarks,upper_body_only, 'R')
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            # 右手
            if right_hand_landmarks is not None:
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, right_hand_landmarks)
                # 描画
                debug_image = draw_hands_landmarks(time,debug_image, cx, cy, right_hand_landmarks,upper_body_only, 'L')
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            time = time + 1
            print(time)
            # キー処理(ESC：終了) #################################################
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break

		    # update the new frame in the frame dictionary
            frameDict[rpiName] = frame

		    # if current time *minus* last time when the active device check
		    # was made is greater than the threshold set then do a check
            if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
			    # loop over all previously active devices
                for (rpiName, ts) in list(lastActive.items()):
				    # remove the RPi from the last active and frame
				    # dictionaries if the device hasn't been active recently
                    if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                        print("[INFO] lost connection to {}".format(rpiName))
                        lastActive.pop(rpiName)
                        frameDict.pop(rpiName)

		    # set the last active check time as current time
            lastActiveCheck = datetime.now()

		    ##############################################################
            cv.imshow('MediaPipe Holistic Demo', debug_image)
			## TERMINA
    ##Creating an array for plot
    if len(R_hand_landmarks) > 0 or len(L_hand_landmarks):
        R_hand_landmarks_transpose = np.transpose(R_hand_landmarks)
        L_hand_landmarks_transpose = np.transpose(L_hand_landmarks)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Trayectory Space of Hand') 
        for k in range(len(R_hand_landmarks_transpose)):
            if k > 1:
                point = R_hand_landmarks_transpose[k]
                x = []
                y = []
                for i in range(len(point)):
                    x.append(point[i][0])
                    y.append(point[i][1])
                ax2.plot(x, y)  
        for k in range(len(L_hand_landmarks_transpose)):
            if k > 1:
                point = L_hand_landmarks_transpose[k]
                x = []
                y = []
                for i in range(len(point)):
                    x.append(point[i][0])
                    y.append(point[i][1])
                ax1.plot(x, y)
        plt.savefig('trajectory_space_of_hand.jpg')
        np.savetxt('R_hand_landmarks_transpose.txt', R_hand_landmarks_transpose, delimiter=",", newline = "\n", fmt="%s")
        np.savetxt('L_hand_landmarks_transpose.txt', L_hand_landmarks_transpose, delimiter=",", newline = "\n", fmt="%s")
    else:
        print('No se encontraron signos')
    cv.destroyAllWindows()
        



if __name__ == '__main__':
    main()

