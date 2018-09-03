import SingleFeatureNeuralNetworks, NeuralNetworks

def get_model(keyword):
    if(keyword == 'text'):
        return SingleFeatureNeuralNetworks.End2EndTextNN
    if(keyword == 'audio'):
        return SingleFeatureNeuralNetworks.End2EndAudioNN
    if(keyword == 'images'):
        return SingleFeatureNeuralNetworks.End2EndVideoNN
    if(keyword == 'microexpressions'):
        return SingleFeatureNeuralNetworks.End2EndMicroexpressionNN
    if(keyword == 'multimodalconcat'):
        return NeuralNetworks.MultimodalConcatNN
    if(keyword == 'multimodalhadamard'):
        return NeuralNetworks.MultimodalHadamardNN
