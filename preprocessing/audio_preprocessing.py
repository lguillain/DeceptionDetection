import os
import numpy as np
import pandas as pd
import pickle as pkl

def extract_audio(name, in_path, out_path):
    OUT_PATH = out_path+name[:-4]+".wav"
    COMMAND = "ffmpeg -i "+in_path+" -f wav -ab 192000 -vn "+OUT_PATH
    os.system(COMMAND)
    print('[INFO] Get Audio from Video')
    os.system('sox '+OUT_PATH+' -n noiseprof '+out_path+name[:-4]+'_noise.prof')
    os.system("sox "+OUT_PATH+" "+out_path+name[:-4]+"_clean.wav noisered "+out_path+name[:-4]+"_noise.prof 0.21")

BASE = '..'
BASE_PATH = BASE+'/data/Real-life_Deception_Detection_2016/Clips'
CONFIG = BASE+'/preprocessing/openSMILE-2.1.0/config/IS13_ComParE.conf'

#Change if filestructure is different, in our dataset split between truthful and deceptive
Truth = os.listdir(BASE_PATH+'/Truthful')
Deception = os.listdir(BASE_PATH+'/Deceptive')

files_truth = list(zip([BASE_PATH+'/Truthful/'+file for file in Truth], Truth))
files_deception = list(zip([BASE_PATH+'/Deceptive/'+file for file in Deception], Deception))

print('[INFO] Now starting to extract audio from video files')
os.system('mkdir '+BASE+'/data/Audio')
for path, name in files_truth:
    extract_audio(name, path, BASE+"/data/Audio/")

for path, name in files_deception:
    extract_audio(name, path, BASE+"/data/Audio/")

print('[INFO] Extraction features using openSmile')
for path, name in files_truth:
    PATH = BASE+"/data/Audio/"+name[:-4]+"_clean.wav"
    os.system("SMILExtract -C "+ CONFIG+" -I "+PATH+" -O "+BASE+"/data/Audio/"+name[:-4]+".csv")

for path, name in files_deception:
    PATH = BASE+"/data/Audio/"+name[:-4]+"_clean.wav"
    os.system("SMILExtract -C "+CONFIG+" -I "+PATH+" -O "+BASE+"/data/Audio/"+name[:-4]+".csv")


print('Parsing the data & extracting relevant features')
started = False
for path, name in files_truth:
    data = np.genfromtxt(BASE+"/data/Audio/"+name[:-4]+".csv", delimiter=',', comments='@')
    data = data[~np.isnan(data)]

    if(started):
        old_data = np.vstack((old_data, data))
    else:
        old_data = data
        started = True

old_data_truth = old_data

Y = np.ones(len(old_data))

started = False
for path, name in files_deception:
    data = np.genfromtxt(BASE+"/data/Audio/"+name[:-4]+".csv", delimiter=',', comments='@')
    data = data[~np.isnan(data)]
    if(started):
        old_data = np.vstack((old_data, data))
    else:
        old_data = data
        started = True

X = np.vstack((old_data_truth, old_data))
Y = np.vstack((np.ones((len(old_data_truth),1)),np.zeros((len(old_data),1))))

print('Saving matrix format')
filehandler = open(BASE+"/data/Audio_X.pkl","wb")
pkl.dump(X,filehandler)
filehandler.close()

filehandler = open(BASE+"/data/Audio_Y.pkl","wb")
pkl.dump(Y,filehandler)
filehandler.close()

print('Saving pickle format')
name = [y for (x,y) in files_truth+files_deception]
labels = ['Truthful']*len(files_truth)+['Deceptive']*len(files_deception)

audio_data = pd.DataFrame([name, list(X), labels]).T
audio_data.columns = ['File', 'Audio', 'Label']
audio_data.head()
audio_data.set_index('File').to_pickle(BASE+'/data/Audio_Dataset.pkl')
os.system('rm -r '+BASE+'/data/Audio/')
