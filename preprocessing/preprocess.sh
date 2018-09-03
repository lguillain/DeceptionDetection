#!/bin/bash
python text_preprocessing.py
echo "Finished preprocessing text"
python audio_preprocessing.py
echo "Finished audio preprocessing"
python generate_audio_model.py
echo "Build pretrained audio model"
python video_preprocessing.py
echo "Finished video to image preprocessing"
python image_dataset_creation.py
echo "Finished creating image dataset"
rm -r ../data/Frames
