# USAGE
# python server.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2

# import the necessary packages
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
from sklearn.manifold import TSNE


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

#################
###################
###################

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=500)
    parser.add_argument("--height", help='cap height', type=int, default=600)

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

    args = parser.parse_args()

    return args

L_hand_landmarks = []
R_hand_landmarks = []

def main():
    
    lastActive = {}
    lastActiveCheck = datetime.now()

    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

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
        image = frame
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
            debug_image = draw_pose_landmarks(debug_image, pose_landmarks,
                                              upper_body_only)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        # Hands ###############################################################
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks

        # 左手
        if left_hand_landmarks is not None:
            # 手の平重心計算 (Calcular centor de gravedad de la palma izquierda)
            cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
            # Calcular el rectangulo delimitador
            brect = calc_bounding_rect(debug_image, left_hand_landmarks)
            #Dibujar
            debug_image = draw_hands_landmarks(time,debug_image, cx, cy,
                                               left_hand_landmarks,
                                               upper_body_only, 'R')
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        # 右手
        if right_hand_landmarks is not None:
            # 手の平重心計算
            cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
            # 外接矩形の計算
            brect = calc_bounding_rect(debug_image, right_hand_landmarks)
            # 描画
            debug_image = draw_hands_landmarks(time,debug_image, cx, cy,
                                               right_hand_landmarks,
                                               upper_body_only, 'L')
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
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
    cv.destroyAllWindows()

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

    print('llego')
    if(len(R_hand_landmarks) > 0):
        R_hand_landmarks_t = np.transpose(R_hand_landmarks_transpose)
        R_hand_landmarks_t = np.delete(R_hand_landmarks_t,0,1)
        R_hand_landmarks_t = np.delete(R_hand_landmarks_t,0,1)
        np.savetxt('R_hand_landmarks_transpose.txt', R_hand_landmarks_t, delimiter=",", newline = "\n", fmt="%s")
        trajectory_space_factorization(R_hand_landmarks_t)
    print('llego2')
    if(len(L_hand_landmarks) > 0):
        L_hand_landmarks_t = np.transpose(L_hand_landmarks_transpose)
        L_hand_landmarks_t = np.delete(L_hand_landmarks_t,0,1)
        L_hand_landmarks_t = np.delete(L_hand_landmarks_t,0,1)
        np.savetxt('L_hand_landmarks_transpose.txt', L_hand_landmarks_t, delimiter=",", newline = "\n", fmt="%s")
        trajectory_space_factorization(L_hand_landmarks_t)
    print('t')




def trajectory_space_factorization(points):
    nFeatures = 21
    W = np.zeros((len(points), nFeatures, 2))

    for index in range(len(points)):
        point_per_frame = points[index]
        for j in range(len(point_per_frame)):
            W[index][j][0] = point_per_frame[j][0] 
            W[index][j][1] = point_per_frame[j][1] 
    nFrames = len(points)
    W_x = W[:,:,0]
    W_y = W[:,:,1]
    W = np.zeros((2*nFrames, nFeatures))
    W[:nFrames, :] = W_x
    W[nFrames:2*nFrames, :] = W_y
    w_bar = W - np.mean(W, axis=1)[:, None]
    w_bar = w_bar.astype('float32')

    u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
    s = np.diag(s_)[:3, :3]
    u = u[:, 0:3]
    v = v[0:3, :]

    S_cap = np.dot(np.sqrt(s), v)
    R_cap = np.dot(u, np.sqrt(s))

    number_of_frame = nFrames

    R_cap_i = R_cap[0:number_of_frame, :]
    R_cap_j = R_cap[number_of_frame:2 * number_of_frame, :]

    A = np.zeros((2 * number_of_frame, 6))
    i = 0
    for i in range(number_of_frame):
        A[2 * i, 0] = (R_cap_i[i, 0] ** 2) - (R_cap_j[i, 0] ** 2)
        A[2 * i, 1] = 2 * ((R_cap_i[i, 0] * R_cap_i[i, 1]) - (R_cap_j[i, 0] * R_cap_j[i, 1]))
        A[2 * i, 2] = 2 * ((R_cap_i[i, 0] * R_cap_i[i, 2]) - (R_cap_j[i, 0] * R_cap_j[i, 2]))
        A[2 * i, 3] = (R_cap_i[i, 1] ** 2) - (R_cap_j[i, 1] ** 2)
        A[2 * i, 5] = (R_cap_i[i, 2] ** 2) - (R_cap_j[i, 2] ** 2)
        A[2 * i, 4] = 2 * ((R_cap_i[i, 2] * R_cap_i[i, 1]) - (R_cap_j[i, 2] * R_cap_j[i, 1]))

        A[2 * i + 1, 0] = R_cap_i[i, 0] * R_cap_j[i, 0]
        A[2 * i + 1, 1] = R_cap_i[i, 1] * R_cap_j[i, 0] + R_cap_i[i, 0] * R_cap_j[i, 1]
        A[2 * i + 1, 2] = R_cap_i[i, 2] * R_cap_j[i, 0] + R_cap_i[i, 0] * R_cap_j[i, 2]
        A[2 * i + 1, 3] = R_cap_i[i, 1] * R_cap_j[i, 1]
        A[2 * i + 1, 4] = R_cap_i[i, 2] * R_cap_j[i, 1] + R_cap_i[i, 1] * R_cap_j[i, 2]
        A[2 * i + 1, 5] = R_cap_i[i, 2] * R_cap_j[i, 2]
    U, SIG, V = np.linalg.svd(A, full_matrices=False)
    v = (V.T)[:, -1]

    QQT = np.zeros((3, 3))

    QQT[0, 0] = v[0]
    QQT[1, 1] = v[3]
    QQT[2, 2] = v[5]

    QQT[0, 1] = v[1]
    QQT[1, 0] = v[1]

    QQT[0, 2] = v[2]
    QQT[2, 0] = v[2]

    QQT[2, 1] = v[4]
    QQT[1, 2] = v[4]

    Q = nearestPD(QQT)

    R = np.dot(R_cap, Q)

    Q_inv = np.linalg.inv(Q)

    S = np.dot(Q_inv, S_cap)
    print(S)
    print(S.shape)

    X = S[0, :]
    Y = S[1, :]
    Z = S[2, :]


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
    RS = 123    
    fashion_tsne = TSNE(random_state=RS).fit_transform(S)
    print('t-SNE done!')
    print(fashion_tsne.shape)
    print(fashion_tsne)


def nearestPD(A):

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


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
        # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

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




if __name__ == '__main__':
    main()