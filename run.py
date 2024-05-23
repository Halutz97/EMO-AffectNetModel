import argparse
import numpy as np
import pandas as pd
import os
import time
from scipy import stats
from functions import sequences
from functions import get_face_areas
from functions.get_models import load_weights_EE, load_weights_LSTM

import pickle

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='video/', help='Path to all videos')
parser.add_argument('--path_save', type=str, default='report/', help='Path to save the report')
parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
parser.add_argument('--path_FE_model', type=str, default='models/EmoAffectnet/weights_0_66_37_wo_gl.h5',
                    help='Path to a model for feature extraction')
parser.add_argument('--path_LSTM_model', type=str, default='models/LSTM/RAVDESS_with_config.h5',
                    help='Path to a model for emotion prediction')

args = parser.parse_args()

def pred_one_video(path):
    print("We're here!")
    start_time = time.time()
    label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    detect = get_face_areas.VideoCamera(path_video=path, conf=args.conf_d)
    dict_face_areas, total_frame = detect.get_frame()
    # with open('face_areas.pkl', 'wb') as file:
        # pickle.dump(dict_face_areas, file)
    name_frames = list(dict_face_areas.keys())
    face_areas = list(dict_face_areas.values())
    print("Number of frames after sampling: ", len(name_frames))
    # Take only the first 8 elements of name_frames and face_areas
    # name_frames = name_frames[:8]
    # face_areas = face_areas[:8]
    EE_model = load_weights_EE(args.path_FE_model)
    LSTM_model = load_weights_LSTM(args.path_LSTM_model)
    # print()
    # print("So far so good")
    # print()
    features = EE_model(np.stack((face_areas)))
    # Features are tensor of 512 elements for each frame
    # Let's inspect features
    # print("Type of features: ", type(features[0]))
    print("Shape of features: ", features.shape)
    # print("features: ", features[0])
    seq_paths, seq_features = sequences.sequences(name_frames, features, win=10, step=4)
    # let's inspect seq_paths and seq_features
    # print("Type of seq_paths: ", type(seq_paths))
    print("Number of steps/windows: ", len(seq_paths))
    # print("seq_paths: ", seq_paths)
    # print("Type of seq_features: ", type(seq_features))
    # print("Shape of seq_features: ", len(seq_features))
    # print("seq_features: ", seq_features)
    # print("Type of seq_features[0]: ", type(seq_features[1]))
    # print("Shape of seq_features[0]: ", len(seq_features[1]))
    # Let's go another level deeper
    # print("Type of seq_features[0][0]: ", type(seq_features[1][1]))
    # print("Shape of seq_features[0][0]: ", len(seq_features[1][1]))
    # Let's go another level deeper
    # print("Type of seq_features[0][0][0]: ", type(seq_features[1][1][0]))
    # Print it
    # print("seq_features[0][0][0]: ", seq_features[1][0][0])



    # return 0
    pred = LSTM_model(np.stack(seq_features)).numpy()
    all_pred = []
    all_path = []
    for id, c_p in enumerate(seq_paths):
        c_f = [str(i).zfill(6) for i in range(int(c_p[0]), int(c_p[-1])+1)]
        c_pr = [pred[id]]*len(c_f)
        all_pred.extend(c_pr)
        all_path.extend(c_f)    
    m_f = [str(i).zfill(6) for i in range(int(all_path[-1])+1, total_frame+1)] 
    m_p = [all_pred[-1]]*len(m_f)
    
    df=pd.DataFrame(data=all_pred+m_p, columns=label_model)
    df['frame'] = all_path+m_f
    df = df[['frame']+ label_model]
    df = sequences.df_group(df, label_model)

    # print("Are we here?")
    
    if not os.path.exists(args.path_save):
        print("Let's create a folder to save the report!")
        os.makedirs(args.path_save)
        
    filename = os.path.basename(path)[:-4] + '.csv'
    df.to_csv(os.path.join(args.path_save,filename), index=False)
    end_time = time.time() - start_time

    # Let's inspect pred
    # print("Type of pred: ", type(pred))
    print("Shape of pred: ", pred.shape)
    # print("pred:")
    # print(pred)

    # What happens if we take the argmax?
    calculate_argmax = np.argmax(pred, axis=1)
    # print("Type of my_argmax: ", type(calculate_argmax))
    print("Shape of argmax: ", calculate_argmax.shape)
    print("argmax: ", calculate_argmax)

    # mode_test = stats.mode(np.argmax(pred, axis=1))
    # Let's inspect mode_test
    # print("Type of mode_test: ", type(mode_test))
    # print("Shape of mode_test: ", mode_test.shape)
    # print("mode_test: ", mode_test)
    # Let's get the mode and count from mode_test
    # print("mode_test.mode: ", mode_test.mode)
    # print("mode_test.count: ", mode_test.count)

    # return 0
    mode_result = stats.mode(np.argmax(pred, axis=1))
    if mode_result.mode.shape == ():
        # Scalar scenario: mode is a single value scalar
        mode = mode_result.mode
    else:
        # Array scenario: mode is an array
        mode = mode_result.mode[0]
    
    print('Report saved in: ', os.path.join(args.path_save,filename))
    print('Predicted emotion: ', label_model[mode])
    print('Lead time: {} s'.format(np.round(end_time, 2)))
    print()

def pred_all_video():
    path_all_videos = os.listdir(args.path_video)
    for id, cr_path in enumerate(path_all_videos):
        print('{}/{}'.format(id+1, len(path_all_videos)))
        pred_one_video(os.path.join(args.path_video,cr_path))
        
        
if __name__ == "__main__":
    # pred_all_video()
    # pred_one_video(r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST\1001_IEO_ANG_HI.mp4")
    pred_one_video(r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST_MP4\1003_IEO_SAD_HI.mp4")
    