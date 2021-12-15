from torch import nn
import torch
import math 

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class LinearWithChannel(nn.Module):
    """
    https://github.com/pytorch/pytorch/issues/36591
    """
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(channel_size, input_size, output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size, 1, output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        b, ch, r  = x.size()
        x = x.view(ch, b, r)
        x = torch.bmm(x, self.weight) + self.bias
        return x.view(b, ch, -1)


class streamModule(nn.Module):
    def __init__(self, input_size = 64, output_size = 64, channel_size = 3):
        super(streamModule, self).__init__()  
        self.lstm = nn.LSTM(input_size = 960,hidden_size=32,num_layers=3, \
                                  bias=True,batch_first=True,dropout=0.5,bidirectional=False)
        self.ffn = LinearWithChannel(input_size, output_size, channel_size)
    def forward(self, stream, x):
        stream, _  = self.lstm( stream )          # # 这里直接用最后一层的输出就不需要max pooling了 -> bs, 3, 32
        x = x.unsqueeze(1).repeat(1,3,1)
        stream = torch.cat([stream, x], axis=2)   # bs * 3 * 64
        stream = self.ffn( stream )                 # bs * 3 * 64
        attw = torch.exp(stream) / torch.sum( torch.exp(stream) , axis = 2).unsqueeze(-1)
        attw = torch.softmax(attw, dim = 1)
        return torch.sum(stream * attw, axis=1)
    


class DeepTransport(nn.Module):
    def __init__(self, convChannel=4, convks = 3, r=3):
        super(DeepTransport, self).__init__()  
        self.fnn1 = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.conv = nn.Sequential(default_conv(1, convChannel, convks), nn.ReLU())  # bn
        self.r = r
        self.upstream = streamModule()
        self.downstream = streamModule()

        self.outfnn = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, 1))  # do not add normalization
        
    def forward(self, up, down, pred):
        bs, r, h, w = up.shape
        pred = self.fnn1( pred )     #  [10, 32]                     
        
        # 需要共享conv
        up = self.conv( up.view(-1, 1, h, w) )           # bs * r, 4, 40, 6
        up = up.view(bs, r, -1)                         # bs, r, 960
        lstmoutUp = self.upstream( up, pred )
        
        down = self.conv( down.view(-1, 1, h, w) )           # bs * r, 4, 40, 6
        down = up.view(bs, r, -1)                         # bs, r, 960
        lstmoutDown = self.upstream( down, pred )
        
#         print( pred.shape, lstmoutUp.shape, lstmoutDown.shape )  # torch.Size([10, 32]) torch.Size([10, 64]) torch.Size([10, 64])
        out = torch.cat([  pred, lstmoutUp, lstmoutDown  ], axis=-1)
        out = self.outfnn(out)
        
        return out
    