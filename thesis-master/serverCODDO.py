##Libraries
"""
    This script extracts video frames from downloaded YouTube videos.
    Video frames are extracted at a rate of N frames every second and rescaled to the
    given size (224x224 by default).
"""
import os
import glob
import math
import cv2
import argparse
import os
import glob
import argparse
import pytictoc
import numpy as np
from PIL import Image
from keras import preprocessing
from keras.applications.vgg16 import preprocess_input
import common
from create_neural_net_model import create_cnn_model
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import chain
from create_neural_net_model import create_neural_net_model

##Extract video frames from a camara web service
##Video frames are extracted at a rate of N frames every second and rescaled to the
##given size (224x224 by default).

frames = [] 
framesXCNN = []
# 0 maps to 'P' (speaking), 1 maps to 'S' (signing), 2 maps to 'n' (other) 
CLASS_MAP = ['P', 'S', 'n']

def extract_video_frames(input_path, resize_shape, output_fps, max_frames_per_video=999999, do_delete_processed_videos=False):
    """
        extracts video frames from a video at a rate of N frames every second (determined by output_fps).
        The video frames are rescaled to the specified frame size.
    """

    # input path must have a file mask
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, '*.*')
        
    # go through each input video
    listing = glob.glob(input_path)
    print('Processing %d video(s)...' % len(listing))

    for file in listing:
        if os.path.isfile(file):
            video = cv2.VideoCapture(file)

            # compute the frame read step based on the video's fps and the output fps
            orig_framerate = video.get(cv2.CAP_PROP_FPS)
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            read_step = math.ceil(orig_framerate / output_fps)

            print('Extracting video frames from %s ...   (%dx%d, %f fps, %d frames)' % (file,
                     int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    orig_framerate, total_frames))
                    
            frame_count = 0
            save_count = 0
            while video.isOpened():
  
                if save_count > max_frames_per_video:
                    print(save_count, max_frames_per_video)
                    break

                success, image = video.read()
                if video.read()[0] is False:
                    print("no leyo el video")
                    break
                if frame_count % read_step == 0:         # save every Nth frame
                    if video.read()[1] is not None:
                        image = cv2.resize(video.read()[1], resize_shape, interpolation = cv2.INTER_AREA)
                        frames.append([str(int(frame_count)),image])
                    save_count += 1
                frame_count += 1

            print('      ...saved %d frames' % save_count)
            video.release()
            print('done')

            if do_delete_processed_videos:
                os.remove(file)


##	python3 generate_CNN_features.py --input=sld/frames --output=sld/frames_cnnfc1 --groundtruth=sld/groundtruth.txt  --fc1_layer=True

def generate_CNN_features( input_file_mask, cnn_model, output_path, groundtruth_file=""):
    # groundtruth data?
    gt = {}
    have_groundtruth_data = False
    if len(groundtruth_file) > 0:
        try:
            # open and load the groundtruth data
            print('Loading groundtruth data...')
            with open(groundtruth_file, 'r') as gt_file:
                gt_lines = gt_file.readlines()
            for gtl in gt_lines:
                gtf = gtl.rstrip().split(' ')
                if len(gtf) == 3:                   # our groundtruth file has 3 items per line (video ID, frame ID, class label)
                    gt[(gtf[0], int(gtf[1]))] = gtf[2]
            print('ok\n')
            have_groundtruth_data = True
        except:
            pass

    tt = pytictoc.TicToc()

    tt.tic()
    id = 0
    for image_j in frames:
            frame_id = image_j[0]
            skip_frame = False
            try:
                skip_frame = True if have_groundtruth_data and gt[(video_i, frame_id)] == '?' else False
            except:
                pass    # safest option is not to skip the frame

            if skip_frame:
                print("x", end='', flush=True)
            else:
                # load the image and convert to numpy 3D array
                img_pil = preprocessing.image.array_to_img(image_j[1])
                img = np.array(preprocessing.image.array_to_img(img_pil))

                # Note that we don't scale the pixel values because VGG16 was not trained with normalised pixel values!
                # Instead we use the pre-processing function that comes specifically with the VGG16
                X = preprocess_input(img)

                X = np.expand_dims(X, axis=0)       # package as a batch of size 1, by adding an extra dimension

                # generate the CNN features for this batch
                print(".", end='', flush=True)
                X_cnn = cnn_model.predict_on_batch(X)

                # save to array
                framesXCNN.append([frame_id,X_cnn])
            id = id + 1    
    tt.toc()
    print('\n\nReady')
    
    print(framesXCNN)
                

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


def test_one_file(path_to_videos, video_id, groundtruth_file, timesteps, image_data_shape, video_data_shape, rnn_input_shape, include_cnn_fc1_layer, model_weights_file, output_path):

    # load the top RNN part of the model, without the convolutional base
    model = create_neural_net_model(image_data_shape, video_data_shape, rnn_input_shape,
            include_convolutional_base=False, include_cnn_fc1_layer=include_cnn_fc1_layer, include_top_layers=True, rnn_model_weights_file=model_weights_file)

    # open and load the groundtruth datass
    print('aqui')
    print('Loading groundtruth data...')
    gt = pd.read_csv(groundtruth_file, delim_whitespace=True, header=None, names=['video_id', 'frame_id', 'gt'])
    print(gt)
    print('ok\n')

    # the video file to be processed
    video_folder = os.path.join(path_to_videos, video_id)

    # select the groundtruth rows for this video
    print('Processing video {} ...'.format(video_id))
    gts = gt.loc[gt['video_id'] == video_id]

    # get all the frames for this video
    frame_list = os.listdir(video_folder)


    cnn_files = []
    gt_labels = []
    pred_labels = []
    frame_numbers = []

    # go through the sampled video frames for which we have CNN features...
    for frame_file in framesXCNN:
        frame_num = int(frame_file[0])
        
        # get groundtruth value
        gt_label = '?'
        if not gts.loc[gts['frame_id'] == frame_num].empty:
            rec = gts.loc[gts['frame_id'] == frame_num]
            gt_label = rec['gt'].values[0]
        print(gt_label, end='', flush=True)

        frame_numbers.append(frame_num)
        cnn_files.append(frame_file[1])
        gt_labels.append(gt_label)
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
            dt = np.load(cnn_files[ndx])
            print(len(dt['X']))
            dt = np.array(dt['X']) 
   # has shape (timesteps, CNN feature vector length)
            X.append(dt[0, ...])
        X = np.array(X)
        # package the input data as a batch of size 1
        X = np.expand_dims(X, axis=0)       # a batch of 1, adding an extra dimension
        # process...
        answer = model.predict(X)
        
        print(answer)
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
    # visualise the groundtruth & the predictions
    fig1 = visualise_labels(gt_labels, 'groundtruth for video {}'.format(video_id))
    if output_path:
        fig1.savefig(os.path.join(output_path, video_id + '_gt.eps'))
    fig2 = visualise_labels(pred_labels, 'predictions for video {}'.format(video_id))
    if output_path:
        fig2.savefig(os.path.join(output_path, video_id + '_pred.eps'))

    # plot the probabilities
    fig3 = plt.figure()
    plt.plot(pred_probs[:, 0], color='green')
    plt.plot(pred_probs[:, 1], color='red')
    plt.plot(pred_probs[:, 2], color='orange')
    plt.show()
    if output_path:
        fig3.savefig(os.path.join(output_path, video_id + '_prob.eps'))    

    # save results to file
    if output_path:
        results_file = open(os.path.join(output_path, video_id+'.txt'), 'w')
        results_file.write('video_id,frame_number,groundtruth_label,predicted_label,predicted_label_probability,probability_P,probability_S,probability_n,mismatch_flag\n')
        for k in range(len(frame_numbers)):
            results_file.write('%s,%d,%s,%s,%f,%f,%f,%f,%s\n' % (video_id, frame_numbers[k], gt_labels[k], pred_labels[k], pred_prob[k], 
                                pred_probs[k,0], pred_probs[k,1], pred_probs[k,2], ' ' if gt_labels[k] == pred_labels[k] else '*WRONG*'))
        results_file.close()

        # save results to file
    if output_path:
        results_file = open(os.path.join(output_path, video_id+'_signing.txt'), 'w')
        results_file.write('video_id,frame_number,predicted_label,\n')
        for k in range(len(frame_numbers)):
            if(pred_labels[k] == "S"):
                results_file.write('%s,%d,%s\n' % (video_id, frame_numbers[k], pred_labels[k]))
        results_file.close()

    print('\nready')




if __name__ == "__main__":
    argparser  =argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input folder containing the downloaded YouTube videos. Can contain a file mask.", default="")
    argparser.add_argument("--fps", help="The rate at which frames will be extracted", default=5)
    argparser.add_argument("--max-frames", help="Maximum number of frames extracted for each individual video", default=2000)
    argparser.add_argument("--imwidth", help="Extracted frames wil be resized to this width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Extracted frames wil be resized to this height (in pixels)", default=224)
    argparser.add_argument("--del-videos", help="Delete each video once frames have been extracted from it", default=False)
    argparser.add_argument("--fc1_layer", help="Include the first fully-connected layer (fc1) of the CNN", default=True)
    argparser.add_argument("--mask", help="The file mask to use for the video frames of the downloaded YouTube videos", default="*.jpg")
    argparser.add_argument("--groundtruth", help="If groundtruth is available, then we load the file in order to only process video frames which have been labelled.", default="")
    argparser.add_argument("--output", help="Output path where the results and figures will be saved to", default="")
    argparser.add_argument("--timesteps", help="Timesteps used in the RNN model. Will depend on the timesteps of the trained RNN model.", default=20)
    argparser.add_argument("--model", help="File path and filename for the trained RNN model weights. File name should be *.h5", default="")
    argparser.add_argument("--video_id", help="The ID of the video to process. Should be a sub-folder of the path given by parameter 'videos'", default="")

    args = argparser.parse_args()

    if not args.input or not args.model or not args.video_id:
        argparser.print_help()
        exit()

    extract_video_frames(input_path=args.input, output_fps=int(args.fps), 
            max_frames_per_video=int(args.max_frames), resize_shape=(int(args.imwidth), 
            int(args.imheight)), do_delete_processed_videos=args.del_videos)

    image_data_shape = (args.imwidth, args.imheight, 3)   # width, height, channels
    model = create_cnn_model(image_data_shape, include_fc1_layer=args.fc1_layer)

    generate_CNN_features(input_file_mask=args.mask, cnn_model=model, output_path=args.output, groundtruth_file=args.groundtruth)

    args.timesteps = int(args.timesteps)

    image_data_shape = (args.imwidth, args.imheight, 3)                         # image width, image height, channels
    video_clip_data_shape = (args.timesteps, args.imwidth, args.imheight, 3)    # timesteps, image width, image height, channels
    rnn_input_shape = (args.timesteps, 4096) if args.fc1_layer else (args.timesteps, 7, 7, 512)    # timesteps, CNN features width, CNN features height, CNN features channels

    t = pytictoc.TicToc()
    t.tic()

    print("Start ")
    test_one_file( video_id=args.video_id, groundtruth_file=args.gt, timesteps=int(args.timesteps),
                image_data_shape=image_data_shape, video_data_shape=video_clip_data_shape, rnn_input_shape=rnn_input_shape, include_cnn_fc1_layer=args.fc1_layer,
                model_weights_file=args.model, output_path=args.output)

    t.toc()