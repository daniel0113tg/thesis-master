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

##Extract video frames from a camara web service
##Video frames are extracted at a rate of N frames every second and rescaled to the
##given size (224x224 by default).

frames = [] 
framesXCNN = []

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
                        frames.append(str(int(frame_count)),image)
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
                img_array = Image.fromarray(image_j[1])
                img = np.array(preprocessing.image.load_img(img_array))

                # Note that we don't scale the pixel values because VGG16 was not trained with normalised pixel values!
                # Instead we use the pre-processing function that comes specifically with the VGG16
                X = preprocess_input(img)

                X = np.expand_dims(X, axis=0)       # package as a batch of size 1, by adding an extra dimension

                # generate the CNN features for this batch
                print(".", end='', flush=True)
                X_cnn = cnn_model.predict_on_batch(X)

                # save to array
                framesXCNN.append(X_cnn)
            id = id + 1    
    tt.toc()
    print('\n\nReady')
    
    print(framesXCNN)
                

                


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
    argparser.add_argument("--output", help="Path to the output folder where the the CNN features will be extracted to", default="")

    args = argparser.parse_args()

    if not args.input:
        argparser.print_help()
        exit()

    extract_video_frames(input_path=args.input, output_fps=int(args.fps), 
            max_frames_per_video=int(args.max_frames), resize_shape=(int(args.imwidth), 
            int(args.imheight)), do_delete_processed_videos=args.del_videos)

    image_data_shape = (args.imwidth, args.imheight, 3)   # width, height, channels
    model = create_cnn_model(image_data_shape, include_fc1_layer=args.fc1_layer)

    generate_CNN_features(input_file_mask=args.mask, cnn_model=model, output_path=args.output, groundtruth_file=args.groundtruth)
