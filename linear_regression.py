'''

Description: PoC of Linear Regression implementation using SMPC using Pysyft
Author: Ionesio Junior

'''

import torch 
from torch import nn
from torch.autograd import Variable 

import syft as sy
import grid as gr

hook = sy.TorchHook(torch)

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto")


x_data = Variable(torch.Tensor( [ [1.5],
                                  [2.5],
                                  [3.5],
                                  [15.2],
                                  [50.5] ] )).fix_precision().share(alice,bob,crypto_provider=crypto_provider)

y_data = Variable(torch.Tensor( [ [3.0], [5.0], [7.0], [30.4], [101.0] ] )) .fix_precision().share(alice,bob,crypto_provider=crypto_provider)


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim, lr):
        super(LinearRegressionModel, self).__init__() 
        self.linear = nn.Linear(input_dim, output_dim)
        self.weight = torch.tensor([[0.0]],requires_grad=True).fix_precision().share(alice,bob,crypto_provider=crypto_provider)
        self.bias = torch.tensor([0.0], requires_grad=True).fix_precision().share(alice,bob,crypto_provider=crypto_provider)
        self.lr = lr

    def forward(self, x):
        out = self._linear(x,self.weight,self.bias)
        return out

    def _linear(self,input_value, weight,bias=None):
        output = input_value.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    def mse_loss(self,input_value, target,
            size_average=None, reduce=None, reduction='mean'):
        dif = input_value - target
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
print (predicted.get().float_precision())
