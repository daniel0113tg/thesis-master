# import the necessary packages
import os
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import copy
import cv2 as cv
import mediapipe as mp
from utils import CvFpsCalc
import matplotlib.pyplot as plt
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import scipy
from matlab_engine import MatlabSession
import scipy.io as spio
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d


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
TIME_WIN_LEN = 12

L_hand_landmarks = []
R_hand_landmarks = []
L_hand_landmarksTSF = []
R_hand_landmarksTSF = []

reading_left_hand = False
reading_right_hand = False

def main():
    print('Server Running')
    run_server()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=500)
    parser.add_argument("--height", help='cap height', type=int, default=600)

    parser.add_argument('--upper_body_only', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.1)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.1)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args




def run_server():
    print('loading arguments')
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    upper_body_only = args.upper_body_only
    print(args.upper_body_only)
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect
    print('loading Google pipeline model')
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        upper_body_only=upper_body_only,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    lastActive = {}
    lastActiveCheck = datetime.now()
    # FPS measurement module
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    time = 0
    sign_state = 0

    while True:
        display_fps = cvFpsCalc.get()
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')
		# if a device is not in the last active dictionary then it means
		# that its a newly connected device
        if rpiName not in lastActive.keys():
        	print("[INFO] receiving data from {}...".format(rpiName))
		# record the last active time for the device from which we just
		# received a frame
        lastActive[rpiName] = datetime.now()
        image = frame
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        # Face Mesh ###########################################################
        face_landmarks = results.face_landmarks
        if face_landmarks is not None:
            # Calculation of bounding rectangles
            brect = calc_bounding_rect(debug_image, face_landmarks)
            # Drawing
            debug_image = draw_face_landmarks(debug_image, face_landmarks)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        # Pose ###############################################################
        pose_landmarks = results.pose_landmarks
        if pose_landmarks is not None:
            # Calculation of bounding rectangles
            brect = calc_bounding_rect(debug_image, pose_landmarks)
            print(brect)
            # Drawing
            debug_image = draw_pose_landmarks(debug_image, pose_landmarks,
                                              upper_body_only)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            centerCoord = (brect[0]+(brect[2]/2), brect[1]+(brect[3]/2))
            print(centerCoord)


        # Hands ###############################################################
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks

        # Left hand
        if left_hand_landmarks is not None:
            reading_left_hand = True
            # Calculation of the centre of gravity of the palm of the hand
            cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
            # Calculation of bounding rectangles
            brect = calc_bounding_rect(debug_image, left_hand_landmarks)
            # Drawing
            debug_image = draw_hands_landmarks(time, centerCoord ,debug_image, cx, cy,
                                               left_hand_landmarks,
                                               upper_body_only, 'R')
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        else: 
            reading_left_hand = False

        # Right hand
        if right_hand_landmarks is not None:
            reading_right_hand = True
            # Calculation of the centre of gravity of the palm of the hand
            cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
            # Calculation of bounding rectangles
            brect = calc_bounding_rect(debug_image, right_hand_landmarks)
            # Drawing
            debug_image = draw_hands_landmarks(time, centerCoord ,debug_image, cx, cy,
                                               right_hand_landmarks,
                                               upper_body_only, 'L')
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        else:
            reading_right_hand = False

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv.LINE_AA)

        if not reading_right_hand and not reading_left_hand:
            cv.putText(debug_image, "NOT READING SING", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
            if (sign_state == 1):
                sign_state = 2
            else:
                sign_state = 0

        if reading_right_hand or reading_left_hand:
            cv.putText(debug_image, "READING SING", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255 , 0), 2, cv.LINE_AA)
            sign_state = 1

        print('state:',sign_state)

        if sign_state == 2:
            runTSF()
            clean_arrays()
            sign_state = 0
        time = time + 1
        # Key processing (ESC) #################################################
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
        cv.imshow('ServerCODDO Demo', debug_image)
			## TERMINA
    cv.destroyAllWindows()


def runTSF():
    print('R',len(R_hand_landmarksTSF),'W',len(L_hand_landmarksTSF))
    if (len(R_hand_landmarksTSF) > 30 and len(L_hand_landmarksTSF) == 0 ):
        print(1,'R',len(R_hand_landmarksTSF),'W',len(L_hand_landmarksTSF))
        graph2D()
        np.savetxt('R_W.txt', R_hand_landmarksTSF, delimiter=",", newline = "\n", fmt="%s")
        SR = trajectory_space_factorization(R_hand_landmarksTSF)
        graph3D(SR)
        Weights = doTSNE(SR)
        print(Weights)

    
    if (len(L_hand_landmarksTSF) > 30 and len(R_hand_landmarksTSF) == 0):
        print(2,'R',len(R_hand_landmarksTSF),'W',len(L_hand_landmarksTSF))
        
        graph2D()
        #np.savetxt('R_W.txt', R_hand_landmarksTSF, delimiter=",", newline = "\n", fmt="%s")
        #np.savetxt('L_W.txt', L_hand_landmarksTSF, delimiter=",", newline = "\n", fmt="%s")
        SL = trajectory_space_factorization(L_hand_landmarksTSF)
        graph3D(SL)
        Weights = doTSNE(SL)
        print(Weights)
        

    if (len(R_hand_landmarksTSF) > 30 and len(L_hand_landmarksTSF) > 30):
        print(3,'R',len(R_hand_landmarksTSF),'W',len(L_hand_landmarksTSF))
        graph2D()
        R_hand_landmarks.pop(0)
        L_hand_landmarks.pop(0)
        graph2D()
        #np.savetxt('R_W.txt', R_hand_landmarksTSF, delimiter=",", newline = "\n", fmt="%s")
        #np.savetxt('L_W.txt', L_hand_landmarksTSF, delimiter=",", newline = "\n", fmt="%s")
        SLR = trajectory_space_factorization(L_hand_landmarksTSF, R_hand_landmarksTSF)
        graph3D(SLR)
        Weights = doTSNE(SLR)
        print(Weights)
    
def clean_arrays():
    L_hand_landmarks.clear()
    R_hand_landmarks.clear()
    L_hand_landmarksTSF.clear()
    R_hand_landmarksTSF.clear()

def doTSNE(SL = None, SR = None):
    if  SL is not None and SR is None:
        L_tsne = TSNE(n_components=1).fit_transform(SL.transpose())
        return L_tsne
    if SR is not None and SL is None:
        R_tsne = TSNE(n_components=1).fit_transform(SR.transpose())
        return R_tsne
    if SR is not None and SL is not None:
        L_tsne = TSNE(n_components=1).fit_transform(SL.transpose())
        R_tsne = TSNE(n_components=1).fit_transform(SR.transpose())
        return np.vstack((L_tsne, R_tsne))
   
    
def graph2D():
     ##Creating an array for plot
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
    '''
    R_hand_landmarks_t = np.transpose(R_hand_landmarks_transpose)
    L_hand_landmarks_t = np.transpose(L_hand_landmarks_transpose)
    R_hand_landmarks_t = np.delete(R_hand_landmarks_t,0,1)
    R_hand_landmarks_t = np.delete(R_hand_landmarks_t,0,1)
    L_hand_landmarks_t = np.delete(L_hand_landmarks_t,0,1)
    L_hand_landmarks_t = np.delete(L_hand_landmarks_t,0,1)
    np.savetxt('R_hand_landmarks_transpose.txt', R_hand_landmarks_t, delimiter=",", newline = "\n", fmt="%s")
    np.savetxt('L_hand_landmarks_transpose.txt', L_hand_landmarks_t, delimiter=",", newline = "\n", fmt="%s")
    '''

def graph3DTotal():
     ##Creating an array for plot
    R_hand_landmarks_transpose = np.transpose(R_hand_landmarks)
       
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    fig.suptitle('Trayectory Space of Hand 3D') 
    z = []
    for k in range(len(R_hand_landmarks_transpose)):
        if k > 1:
            z.append(k-1)
            point = R_hand_landmarks_transpose[k]
            x = []
            y = []
            for i in range(len(point)):
                x.append(point[i][0])
                y.append(point[i][1])
            ax.plot(z, x, y)  
    plt.show()




def graph3D(S):
    X = S[0, :]
    Y = S[1, :]
    Z = S[2, :]

    print(X)
    pointcloud = np.zeros((X.shape[0], 3))
    pointcloud[:,0] = X
    pointcloud[:,1] = Y
    pointcloud[:,2] = Z


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    r = np.array(X, dtype=np.float)
    s = np.array([float(i) for i in Y])
    t = np.array(Z) * 1.0

    ax.scatter(r,s,zs = t, s=200, label='True Position')

    #pulgar
    ax.plot3D([r[2], r[3]], [s[2], s[3]],[t[2], t[3]], 'b')
    ax.plot3D([r[3], r[4]], [s[3], s[4]],[t[3], t[4]], 'b')

    #indice
    ax.plot3D([r[5], r[6]], [s[5], s[6]],[t[5], t[6]], 'b')
    ax.plot3D([r[6], r[7]], [s[6], s[7]],[t[6], t[7]], 'b')
    ax.plot3D([r[7], r[8]], [s[7], s[8]],[t[7], t[8]],'b')

     #medio
    ax.plot3D([r[9], r[10]], [s[9], s[10]],[t[9], t[10]], 'b')
    ax.plot3D([r[10], r[11]], [s[10], s[11]],[t[10], t[11]], 'b')
    ax.plot3D([r[11], r[12]], [s[11], s[12]],[t[11], t[12]],'b')

    #anular
    ax.plot3D([r[13], r[14]], [s[13], s[14]],[t[13], t[14]], 'b')
    ax.plot3D([r[14], r[15]], [s[14], s[15]],[t[14], t[15]], 'b')
    ax.plot3D([r[15], r[16]], [s[15], s[16]],[t[15], t[16]],'b')

    #meñique
    ax.plot3D([r[17], r[18]], [s[17], s[18]],[t[17], t[18]], 'b')
    ax.plot3D([r[18], r[19]], [s[18], s[19]],[t[18], t[19]], 'b')
    ax.plot3D([r[19], r[20]], [s[19], s[20]],[t[19], t[20]],'b')

    #palma
    ax.plot3D([r[0], r[1]], [s[0], s[1]],[t[0], t[1]], 'b')
    ax.plot3D([r[1], r[2]], [s[1], s[2]],[t[1], t[2]], 'b')
    ax.plot3D([r[2], r[5]], [s[2], s[5]],[t[2], t[5]],'b')
    ax.plot3D([r[5], r[9]], [s[5], s[9]],[t[5], t[9]], 'b')
    ax.plot3D([r[9], r[13]], [s[9], s[13]],[t[9], t[13]], 'b')
    ax.plot3D([r[13], r[17]], [s[13], s[17]],[t[13], t[17]],'b')
    ax.plot3D([r[17], r[0]], [s[17], s[0]],[t[17], t[0]],'b')

    plt.show()




def trajectory_space_factorization(L_hand_landmarksTSF = None, R_hand_landmarksTSF = None):
    # prepare a MATLAB session
    ms = MatlabSession()
    ms.m.cd(os.getcwd())    # ensure MATLAB points to the current folder
    if L_hand_landmarksTSF is not None and R_hand_landmarksTSF is None:
        Wl = build_trajectory_matrix(L_hand_landmarksTSF)
        output_filenameWl = 'Wl.mat'
        spio.savemat(output_filenameWl, {'W' : Wl}, appendmat=False, do_compression=False)
        Reconstruction3DL = ms.m.run_trajectory_space_factorisation(output_filenameWl,"2DL.png")
        SL = np.asarray(Reconstruction3DL)
        return SL

    if R_hand_landmarksTSF is not None and L_hand_landmarksTSF is None:
        Wr = build_trajectory_matrix(R_hand_landmarksTSF)
        output_filenameWr = 'Wr.mat'
        spio.savemat(output_filenameWr, {'W' : Wr}, appendmat=False, do_compression=False)
        Reconstruction3DR = ms.m.run_trajectory_space_factorisation(output_filenameWr,"2DR.png")
        SR = np.asarray(Reconstruction3DR)
        return SR

    if L_hand_landmarksTSF is not None and R_hand_landmarksTSF is not None:
        Wl = build_trajectory_matrix(L_hand_landmarksTSF)
        Wr = build_trajectory_matrix(R_hand_landmarksTSF)
        W = np.hstack((Wl,Wr))
        output_filenameW = 'W.mat'
        spio.savemat(output_filenameW, {'W' : W}, appendmat=False, do_compression=False)

        Reconstruction3D = ms.m.run_trajectory_space_factorisation(output_filenameW,"2DSLR.png")

        SLR = np.asarray(Reconstruction3D)
        return SLR



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
                        centerCoord,
                        image,
                         cx,
                         cy,
                         landmarks,
                         upper_body_only,
                         handedness_str='R',):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    landmark_pointsX = []
    landmark_pointsY = []
    landmark_point.append(time)
    #landmark_pointsX.append(time)
    #landmark_pointsY.append(time)
    landmark_point.append(handedness_str)
    #nan = ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan']

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1) - int(centerCoord[0])
        landmark_y = min(int(landmark.y * image_height), image_height - 1) - int(centerCoord[1])  
        landmark_x_nocentroid = min(int(landmark.x * image_width), image_width - 1)
        landmark_y_nocentroid = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_pointsX.append(landmark_x )
        landmark_pointsY.append(landmark_y)
        landmark_point.append((landmark_x_nocentroid, landmark_y_nocentroid))

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
            L_hand_landmarksTSF.append(landmark_pointsX)
            L_hand_landmarksTSF.append(landmark_pointsY)

        if handedness_str == 'R':
            R_hand_landmarks.append(landmark_point)
            R_hand_landmarksTSF.append(landmark_pointsX)
            R_hand_landmarksTSF.append(landmark_pointsY)


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
        # https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

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

def build_trajectory_matrix(ldmrkpts):
    W = np.array(ldmrkpts)
    return W


def is_trajectory_matrix_complete(W):
    return np.all(~np.isnan(W))


def remove_incomplete_trajectories(W):
    W = W[:, ~np.any(np.isnan(W), axis=0)]
    return W




def display_3d_trajectories(W, figure_name="3D trajectories", **kwargs):
    """
        Extra arguments passed to the function will be passed as arguments o the internal plot function.
        Can also pass 'azim' and 'elev' to set the initial camera viewpoint.
    """
    fig = plt.figure(figure_name)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    if 'azim' in kwargs and 'elev' in kwargs:
        ax.view_init(elev=kwargs['elev'], azim=kwargs['azim'])
        del kwargs['elev']
        del kwargs['azim']
    print(range(0, len(W[0])-1))
    print(W[0][range(0, len(W[0])-1)])
    xs=list(W[0][range(0, len(W[0])-1, 2)])
    print(xs)
    ys=list(W[0][range(1, len(W[0]), 2)])
    zs=range(0, len(W[0]) // 2)

    for col in W:
        ax.plot(xs, ys, zs, zdir='y', **kwargs)
    plt.xlabel('u')
    plt.ylabel('t')
    plt.show()



def do_matrix_completion(W):
    scaler = StandardScaler()
    do_scaling = True
    if do_scaling:
        scaler.fit(W)
        W = scaler.transform(W)
    W = SoftImpute(verbose=True).fit_transform(W)
    if do_scaling:
        W = scaler.inverse_transform(W)

    return W




if __name__ == '__main__':
    main()