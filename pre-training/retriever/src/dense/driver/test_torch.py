import torch

a = torch.randn(1,4)
b = torch.randn(1,4)
c = torch.randn(1,4)

all = [a,b,c]

d = torch.stack(all,dim=0)

res = torch.mean(d,dim=0)

print(res.shape)

print(a)
print(b)
print(c)
print(res)