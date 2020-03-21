import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

# print(tensor)
# print(variable)
t_out = torch.mean(tensor * tensor)
variable_out = torch.mean(variable * variable)

variable_out.backward()
print(t_out)
print(variable_out)
