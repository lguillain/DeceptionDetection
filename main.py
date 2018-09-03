from DatasetHandler import FastMultimodalDataset
from utils.TrainingHelpers import get_trainset, run_model, get_samples, save
import SingleFeatureNeuralNetworks as nn
import numpy as np
import os
import TelegramBot
from utils.GetModel import get_model
#from utils.Plotting import plot_history

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('-cuda', default=False, help='enable the gpu') some things are missing to be cuda compatible
parser.add_argument('-feature', default='text', help='choose model')
parser.add_argument('-epochs', default=25, help='choose number of epochs')

args = parser.parse_args()

if not os.path.isdir('saved_models/{}'.format(args.feature)):
    os.makedirs('saved_models/{}'.format(args.feature))

if 'multi' in args.feature:
    print('[INFO] Using multimodal model')
    #run_model = run_model_multimodal

PATH = 'data/'
dataset = FastMultimodalDataset(root_annotation=PATH+'Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv',
                            root_images=PATH+'images_normalized.npy',
                            root_audio=PATH+'Audio_Dataset.pkl',
                            root_text=PATH+'Text_Dataset.pkl')
sets = get_trainset(30)

learning_rates = [1e-4, 1e-6]
num_epochs = args.epochs
highest_accuracy = []

m = get_model(args.feature)

for dp in [0, 0.1, 0.25, 0.5]:
    for hidden_size in [64, 256, 512, 1024]:
        for learning_rate in learning_rates:
            best_accuracy = 0.44
            history = {'train_loss' : [], 'test_loss': [], 'train_acc' : [], 'test_acc': []}
            title = 'hidden'+str(hidden_size)+'_dropout'+str(dp)+'_lr'+str(learning_rate)

            print('[INFO] hidden layer size: ', hidden_size)
            print('\tdropout: ', dp)
            print('\tlearning rate: ', learning_rate)

            for i, testset in enumerate(sets):
                trainset = sets[:i]+sets[i+1:]
                trainset = [item for sublist in trainset for item in sublist]
                train_loader, test_loader, n_training_samples, n_test_samples = get_samples(dataset,
                                                                                            trainset,
                                                                                            testset, 3)
                model = m(hidden_size=hidden_size, dropout=dp)
                loss_train, accuracy_train, loss_test, accuracy_test = run_model(model, learning_rate, num_epochs,
                          train_loader, n_training_samples,
                          test_loader, n_test_samples, args.feature, cuda=args.cuda)

                history['train_loss'].append(loss_train)
                history['test_loss'].append(loss_test)

                history['train_acc'].append(accuracy_train)
                history['test_acc'].append(accuracy_test)

                mean_acc = np.array(accuracy_test).max()

                if mean_acc > best_accuracy:
                    print('[INFO] Best test accuracy so far: {}'.format(mean_acc))
                    save(model, 'saved_models/{}'.format(args.feature), title+'_accuracy_'+str(mean_acc)[:5])
                    best_accuracy = mean_acc


            print('[INFO] Best mean test accuracy: ', np.array(history['test_acc']).mean(axis=0).max())

            # plot_history(history['test_acc'])
            # plot_history(history['train_acc'])
            # plt.title(title)
            # plt.savefig('Figures/{}'.format(title))

            #TelegramBot.send_msg('Finished Training untrained Embedding Text CNN!')
