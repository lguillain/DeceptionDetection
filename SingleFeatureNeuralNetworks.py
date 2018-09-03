from __future__ import print_function
import torch
import sklearn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from NeuralNetworks import  FinalNN, AudioNN, TextCNN, VideoFrameCNN

class End2EndMicroexpressionNN(nn.Module):
    def __init__(self, dropout=0.5, hidden_size=1024):
        super(End2EndMicroexpressionNN, self).__init__()
        self.final = FinalNN(39, hidden_size, 2, dropout)

    def forward(self, x):
        out = self.final(x)
        return out

class End2EndAudioNN(nn.Module):
    def __init__(self, dropout=0.5, hidden_size=1024):
        super(End2EndAudioNN, self).__init__()
        self.audio = AudioNN()
        self.final = FinalNN(300, hidden_size, 2, dropout)

    def forward(self, x):
        out1 = self.audio(x)
        out2 = self.final(out1)
        return out2

class End2EndTextNN(torch.nn.Module):
    def __init__(self, hidden_size=1024, dp1=0.5, dropout=0.5):
        super(End2EndTextNN, self).__init__()
        self.conv = TextCNN(dp1)

        input_size = 300
        num_classes = 2
        self.through_nn = FinalNN(input_size, hidden_size, num_classes, dp2)

    def forward(self, x):
        rep = self.conv(x)
        out = self.through_nn(rep)
        return out

class End2EndVideoNN(torch.nn.Module):
    def __init__(self, dropout=0.5, hidden_size=1024):
        super(End2EndVideoNN, self).__init__()
        self.conv = VideoFrameCNN()

        input_size = 300
        num_classes = 2
        self.through_nn = FinalNN(input_size, hidden_size, num_classes, dp)

    def forward(self, x):
        rep = self.conv(x)
        out = self.through_nn(rep)
        return out
