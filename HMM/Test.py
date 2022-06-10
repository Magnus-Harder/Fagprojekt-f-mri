#%%
import torch
Softmax = torch.nn.Softmax(0)
Softmax2 = torch.nn.Softmax(1)
T = torch.ones((3,3))
for i in range(3):
    T[i,i] = i+1
T[2,0] = 4
p = torch.ones(3)/3
p2 = torch.tensor([0.2,0.5,0.3])
p3 = torch.tensor([1.,0,0])
p4 = torch.tensor([0,1.,0])
p5 = torch.tensor([0,0,1.])
def lsum(x):
    Max = x.max(1).values
    #return  Maxes + torch.log(torch.exp(x-Maxes).sum())
    return torch.log(torch.exp(x-Max).sum(1)) + Max
Optimized = Softmax(T)

print(sum(Optimized@p))
print(sum(Optimized@p2))
print(sum(Optimized@p3))
print(sum(Optimized@p4))
print(sum(Optimized@p5))
lsum(T+p)