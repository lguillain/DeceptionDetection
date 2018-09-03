import os
import numpy as np
import pandas as pd
import pickle as pkl

BASE = '../data'
BASE_PATH = BASE + '/Real-life_Deception_Detection_2016/Clips'

Truth = os.listdir(BASE_PATH+'/Truthful')
Deception = os.listdir(BASE_PATH+'/Deceptive')

files_truth = list(zip([BASE_PATH+'/Truthful/'+file for file in Truth], Truth))
files_deception = list(zip([BASE_PATH+'/Deceptive/'+file for file in Deception], Deception))

print('Starting to splice frames from videos')
for filename, name in files_deception:
    #splicing
    os.system('mkdir -p '+BASE+'/Frames/'+name[:-4])
    commandSpliceToImage = "ffmpeg -i " + filename + " "+BASE+"/Frames/" + name[:-4] + "/_%03d.jpg"
    os.system(commandSpliceToImage)

for filename, name in files_truth:
    #splicing
    os.system('mkdir -p '+BASE+'/Frames/'+name[:-4])
    commandSpliceToImage = "ffmpeg -i " + filename + " "+BASE+"/Frames/" + name[:-4] + "/_%03d.jpg"
    os.system(commandSpliceToImage)

print('Finished splicing \nStarting to gather metadata')

length = []
start = []
end = []
old = 0
base_path = BASE+"/Frames/"
names = []
list_ = sorted(os.listdir(base_path))
for name in list_:
    try:
        f = base_path + name
        l = len(os.listdir(f))
        length.append(l)
        start.append(old)
        end.append(old+l-1)
        old += l
        names.append(name)
    except:
        print('[ERROR] failed!!!!')
        continue

times = pd.DataFrame([names, length, start, end]).T
times.columns= ['name', 'duration', 'start', 'end']
times.set_index('name').to_csv(BASE+'/video_metadata.csv')
