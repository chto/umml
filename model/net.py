"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, downsample=None, padding=1):
       super(Block, self).__init__()
       self.conv1 = nn.Conv3d(in_channels, channels,3, stride=stride, padding=padding, bias=False)
       self.bn1 = nn.BatchNorm3d(channels)
       self.conv2 = nn.Conv3d(channels, channels,3, stride=1, padding=padding, bias=False)
       self.bn2 = nn.BatchNorm3d(channels)
       self.downsample = downsample
       self.stride = stride

    def forward(self, s):
        residual = s 
        out = self.conv1(s) 
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(s)
        sys.stdout.flush()
        out += residual
        out = F.relu(out)
        return out
class Block1D(nn.Module):
    def __init__(self, in_channels, channels, stride=1, downsample=None, padding=1):
       super(Block1D, self).__init__()
       self.conv1 = nn.Conv2d(in_channels, channels,(3,3), stride=(stride,1), padding=(padding,padding), bias=False)
       self.bn1 = nn.BatchNorm2d(channels)
       self.conv2 = nn.Conv2d(channels, channels,3, stride=1, padding=padding, bias=False)
       self.bn2 = nn.BatchNorm2d(channels)
       if stride !=1 :
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels,
                          kernel_size=1, stride=(stride,1), bias=False),
                nn.BatchNorm2d(channels),
            )
       self.downsample = downsample
       self.stride = stride

    def forward(self, s):
        residual = s 
        out = self.conv1(s) 
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(s)
        sys.stdout.flush()
        out += residual
        out = F.relu(out)
        return out
class ResNet18(nn.Module):
    """
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains num_channels
        """
        super(ResNet18, self).__init__()
        self.params = params
        layers = [2,2,2,2]
        self.num_channels = params.num_channels
        self.inchannels = params.num_channels
        
        self.conv1 = nn.Conv3d(1,self.inchannels, 3, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inchannels)
        self.maxpool = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(Block,self.num_channels,layers[0], stride=2)
        self.layer2 = self._make_layer(Block,2*self.num_channels,layers[1], stride=2)
        self.layer3 = self._make_layer(Block,4*self.num_channels,layers[2], stride=2)
        #self.avgpool = nn.AvgPool2d(4,stride=1)
        self.fc1 = nn.Linear(4*self.num_channels,320)
        self.fcbn1 = nn.BatchNorm1d(320)
        self.fc2 = nn.Linear(320,80)
        self.fcbn2 = nn.BatchNorm1d(80)
        #self.fc3 = nn.Linear(80,34)
        #self.fcbn3 = nn.BatchNorm1d(34)
        self.fc3 = nn.Linear(80,14)
        self.fcbn3 = nn.BatchNorm1d(14)
        self.dropout_rate = params.dropout_rate

        self._initialize_weights()

        #self.dropout_rate = params.dropout_rate
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, num_layers, stride=1):
        downsample = None
        if stride != 1 or self.inchannels != channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.inchannels, channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels),
            )
        layers = []
        layers.append(block(self.inchannels, channels, stride, downsample))
        self.inchannels = channels
        for i in range(1, num_layers):
            layers.append(block(self.inchannels, channels))
        return nn.Sequential(*layers)

    def forward(self, s):
        """
        """
        s = s.view(s.size(0), 1, s.size(1), s.size(2), s.size(3))
        s = self.conv1(s)
        s = self.bn1(s)
        s = F.relu(s)
        s =  self.maxpool(s)
        s = self.layer1(s)
        s = self.layer2(s)
        s = self.layer3(s)
        #s = self.avgpool(s)
        s = s.view(s.size(0),-1)
        s =  F.dropout(F.relu(self.fcbn1(self.fc1(s))),p=self.dropout_rate, training=self.training) 
        s = F.dropout(F.relu(self.fcbn2(self.fc2(s))),
                    p=self.dropout_rate, training=self.training)
        s = self.fcbn3(self.fc3(s))
        return s 
        #s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
        #    p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        #s = self.fc2(s)                                     # batch_size x 3

        #return F.softmax(s, dim=1)


class Fullconnect(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Fullconnect, self).__init__()
        self.num_channels = params.num_channels
        self.minibatchsize = params.batch_size
        
        self.fc1 = nn.Linear(15625,1000)       
        self.fcbn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000,150)       
        self.fcbn2 = nn.BatchNorm1d(150)
        self.fc3 = nn.Linear(150,34)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = F.relu(self.fcbn1(self.fc1(s.view(s.size(0),-1))))
        s = F.relu(self.fcbn2(self.fc2(s)))
        s = self.fc3(s)
        # flatten the output for each image

        # apply 2 fully connected layers with dropout
        #s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
        #    p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return s





class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.minibatchsize = params.batch_size
        #print(type(self.num_channels))
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(25, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        #self.fc1 = nn.Linear(1280, self.num_channels*4)
        #self.fc2 = nn.Linear(self.num_channels*4, 32)       
        self.fc1 = nn.Linear(360,68)       
        self.fcbn1 = nn.BatchNorm1d(68)
        self.fc2 = nn.Linear(68,34)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8
        s = torch.squeeze(s)
        fc1 = nn.Linear(11520,1088) 
        s = fc1(s.view(-1,11520))
        s = s.view(32,34)
        """
        s_clone = torch.ones((self.minibatchsize, 68))
        for i in range(self.minibatchsize):
            s_clone[i] = self.fc1(s[i].view(-1,360))                                     # batch_size x 6
        s = s_clone
        s = self.fcbn1(s)
        s_clone = torch.ones((self.minibatchsize, 34))
        for i in range(self.minibatchsize):
            s_clone[i] = F.dropout(F.relu(self.fc2(s[i])), 
                                   p=self.dropout_rate, training=self.training)
        s = s_clone
        """
        # flatten the output for each image

        # apply 2 fully connected layers with dropout
        #s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
        #    p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return s

##From https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html
def _batch_mahalanobis(L, x):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both L and x.
    """
    # TODO: use `torch.potrs` or similar once a backwards pass is implemented.
    flat_L = L 
    L_inv = torch.stack([torch.inverse(Li.t()) for Li in flat_L]).view(L.shape)
    return torch.mean((x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1))



def loss_fn(outputs, labels, datas):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    #num_examples = datas.size()[0]
    #data_in = torch.cumsum(datas.view((num_examples, -1)),-1)[:,-1]
    #output_in = torch.cumsum(outputs.view((num_examples, -1)), -1)[:,-1]
    #output_in_check = output_in[data_in==0]
    offset=0
    #if torch.cumsum(output_in, 0)[-1]<1E-30:
    #    offset = torch.tensor([1E100])
    #if output_in_check.nelement() != 0:
    #    if torch.cumprod(output_in_check, 0)[-1]!=0:
    #        offset = torch.tensor([1E20])


    x = outputs-labels[:,0,:]
    L = torch.stack([torch.diag(temp) for temp in labels[:,1,:]]) ##labels[:,1,:] is sigma
    #x = outputs-labels[:,:,0]
    #L = torch.stack([torch.diag(temp) for temp in labels[:,:,1]]) ##labels[:,1,:] is sigma
    #print(x.shape)
    #for multiplier in np.linspace(0.001,2,100):
    #    x = 0*outputs+(multiplier-1)*labels[:,0,:]
    #    print(multiplier, _batch_mahalanobis(L, x)+offset)
    #assert False
    return _batch_mahalanobis(L, x)+offset
def smf(outputs, labels):
    return [np.mean(outputs, axis=0), np.mean(labels, axis=0)[0]]
def chisq(outputs, labels):
    x = (outputs - labels[:,0,:])
    chisq2 = np.sum(x**2/labels[:,1,:]**2,axis=1)
    return np.mean(chisq2)
    
    


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    "smf": smf,
    "chisq": chisq,
    # could add more metrics such as accuracy for each token type
}
