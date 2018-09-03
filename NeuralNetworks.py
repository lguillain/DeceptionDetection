from __future__ import print_function
import torch
import sklearn
import torch.nn as nn
import torch.nn.functional as F
import torchvision



#############TEXT##############
def create_emb_layer(weights_matrix):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})

    return emb_layer, num_embeddings, embedding_dim

class TextCNN(torch.nn.Module):

    def __init__(self):
        super(TextCNN, self).__init__()

        #Parameters as taken from paper

        Ks = [3, 5, 8] #dimension of filters
        E = 300 #Embedding dimension
        Nf = 20 #Number of filters
        C = 300 #Number of classes (output)

        self.embedding = nn.Embedding(1522, 300)
        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])
        self.maxpool = torch.nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3800, C)  # a dense layer for classification

    def forward(self, x):
        x = self.embedding(x.type(torch.LongTensor))
        x = x.permute(0, 2, 1).float()
        x = [self.conv_and_max_pool(x, k) for k in self.convs]
        x = torch.cat(x, 1).view(-1, 3800)
        x = self.fc1(x)

        return(x)

    #@staticmethod
    def conv_and_max_pool(self, x, conv):
        """Convolution and global max pooling layer"""
        res = F.relu(self.maxpool(conv(x))).permute(0, 2, 1)
        return res

##########VIDEO

class VideoFrameCNN(torch.nn.Module):
    def __init__(self):
        super(VideoFrameCNN, self).__init__()
        self.CNN = nn.Conv3d(3, 32, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool3d(kernel_size=3)
        self.fc = nn.Linear(1534752, 300)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.CNN(x)
        out = self.maxpool(out)
        out = self.fc(out.view(-1, 1534752))
        return out

#########AUDIO############
class AudioNN(nn.Module):
    """ NN that trains audio features - used for signular model (may remove through nn for complete model)"""
    def __init__(self, base_input_size=6373, input_size=300, hidden_size=64, dropout=0.5, audio_model='saved_models/audio_model'):
        super(AudioNN, self).__init__()
        self.lin = nn.Linear(base_input_size, input_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.audio_model = torch.load(audio_model)

    def forward(self, x):
        en, de = self.audio_model(x)
        en = self.relu(en)
        out = self.dropout(en)
        return out

class EncodingNN(nn.Module):
    """
    NN class used to reduce size and encode audio features
    """

    def __init__(self, input_size=6373, hidden_size=300, num_classes=6373):
        super(EncodingNN, self).__init__()
        self.lin = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        l1 = self.lin(x)
        encoded = self.relu(l1)
        decoded = self.lin2(encoded)
        return l1, decoded

class FinalNN(nn.Module):
    """
    NN going on top of feature dependent Networks
    """

    def __init__(self, input_size, hidden_size=1024, num_classes=2, dropout=0.5):

        super(FinalNN, self).__init__()
        self.lin = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        l1 = self.lin(x)
        r1 = self.relu(l1)
        dp = F.dropout(r1, training=self.training)
        l2 = self.lin2(dp)
        return l2


class MultimodalConcatNN(torch.nn.Module):

    def __init__(self, hidden_size, dropout):
        super(MultimodalConcatNN, self).__init__()
        self.text = TextCNN()
        self.audio = AudioNN()
        self.video = VideoFrameCNN()

        input_size = 300*3+39
        num_classes = 2
        self.through_nn = FinalNN(input_size, hidden_size, num_classes, dropout)

    def forward(self, x):
        text = self.text(torch.Tensor(x['text'].float()))
        audio = self.audio(torch.Tensor(x['audio'].float()))
        video = self.video(torch.Tensor(x['images'].float()))
        multi = torch.cat([text, audio, video, torch.Tensor(x['microexpressions'].float())], 1)
        out = self.through_nn(multi)
        return out

class MultimodalHadamardNN(torch.nn.Module):

    def __init__(self, hidden_size, dropout):
        super(MultimodalHadamardNN, self).__init__()
        self.text = TextCNN()
        self.audio = AudioNN()
        self.video = VideoFrameCNN()

        input_size = 300+39
        num_classes = 2
        self.through_nn = FinalNN(input_size, hidden_size, num_classes, dropout)

    def forward(self, x):
        text = self.text(torch.Tensor(x['text'].float()))
        audio = self.audio(torch.Tensor(x['audio'].float()))
        video = self.video(torch.Tensor(x['images'].float()))
        hadamard = text * audio * video
        print(hadamard.size())
        multi = torch.cat([hadamard, torch.Tensor(x['microexpressions'].float())], 1)
        print(multi.size())
        out = self.through_nn(multi)
        return out
