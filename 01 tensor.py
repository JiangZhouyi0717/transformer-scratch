import torch
print("===Pytorch 环境测试===")
print(f"pytorch版本：{torch.__version__}")
print(f"CUDA可用：{torch.cuda.is_available()}")
print(f"GPU名称：{torch.cuda.get_device_name(0)}")

print("\n===张量基础操作===")
x = torch.randn(3, 4, device="cuda")
print(f"x的形状{x.shape}")
print(f"x的设备{x.device}")

print("\n===张量变形===")
y=x.reshape(4,3)
print(f"变形后的形状{y.shape}")
z=x.T
print(f"x的转置的形状{z.shape}")
first_row=x[0,:]
print(f"第一行{first_row}")
w=x@x.T
print(f"w的形状{w.shape}")

print("\n===自动求导演示===")
a=torch.randn(3,3,device="cuda",requires_grad=True)
b=a.mean()
b.backward()
print(f"a的梯度的形状{a.grad.shape}")
print(f"a的梯度{a.grad}")
