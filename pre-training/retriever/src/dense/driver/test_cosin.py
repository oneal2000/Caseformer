from scipy import spatial
vec1 = [1, 2, 3, 4]
vec2 = [5, 6, 7, 8]
cos_sim = 1 - spatial.distance.cosine(vec1, vec2)
print(cos_sim)


import torch.nn.functional as F
import torch

a = torch.tensor(vec1,dtype=torch.float)
b = torch.tensor(vec2,dtype=torch.float)

res = F.cosine_similarity(a, b, dim=0)
print(res)