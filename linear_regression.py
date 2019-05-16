'''

Description: PoC of Linear Regression implementation using SMPC using Pysyft
Author: Ionésio Junior

'''

import torch 
from torch import nn
from torch.autograd import Variable 

import syft as sy
from syft.workers import WebsocketClientWorker



hook = sy.TorchHook(torch)
kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": False}

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

x_data = Variable(torch.Tensor( [ [1.5],
                                  [2.5],
                                  [3.5],
                                  [15.2],
                                  [50.5] ] )).fix_precision().share(alice,bob,crypto_provider=crypto_provider)
y_data = Variable(torch.Tensor( [ [3.0], [5.0], [7.0], [30.4], [101.0] ] )) # Find way to anonymize labels



class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim, lr):
        super(LinearRegressionModel, self).__init__() 
        self.linear = nn.Linear(input_dim, output_dim)
        self.weight = torch.Tensor(1,1).fix_precision().share(alice,bob,crypto_provider=crypto_provider)
        self.bias = torch.tensor(1).fix_precision().share(alice,bob,crypto_provider=crypto_provider)
        self.lr = lr

    def forward(self, x):
        out = self._linear(x,self.weight,self.bias)
        return out

    def _linear(self,input_value, weight,bias=None):
        output = input_value.matmul(weight)
        if bias is not None:
            output += bias
        ret = output.copy().get().float_precision() # arithmetic operations save result at only one worker (need to fix this)
        return ret

    def mse_loss(self,input_value, target,
            size_average=None, reduce=None, reduction='mean'):
        dif = (input_value - target).fix_precision().share(alice,bob,crypto_provider=crypto_provider)
        ret = (dif * x_data).get().float_precision() # recover J(theta) to perform torch.mean
        if reduction != "none":
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret

    def SGD_step(self, loss ):
        self.weight += (-self.lr * loss).fix_precision().share(alice,bob,crypto_provider=crypto_provider) # update anonymised weights
        
input_dim = 1
output_dim = 1
l_rate = 0.001
model = LinearRegressionModel(input_dim,output_dim,l_rate)

epochs = 2000
for epoch in range(epochs):

    epoch +=1
    
    inputs = x_data
    labels = y_data
    
    outputs = model.forward(inputs)
    loss = model.mse_loss(outputs, labels, reduction="mean")
    if(epoch % 10 == 0):
        print("Epoch ", epoch, "/", epochs, "Loss : ", abs(loss))    
    model.SGD_step(loss)

predicted = model.forward(x_data)
print (predicted)
