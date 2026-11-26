import torch
import gsplat
from gsplat import rasterization

device = "cuda:0"

print("PyTorch:", torch.__version__)
print("gsplat:", gsplat.__version__)
print("GPU:", torch.cuda.get_device_name(0))

# 创建三个三维高斯点
means = torch.tensor([
    [-0.5, 0.0, 3.0],
    [ 0.0, 0.0, 3.0],
    [ 0.5, 0.0, 3.0]
], device=device, requires_grad=True)

# 高斯旋转四元数，wxyz顺序
quats = torch.tensor([
    [1., 0., 0., 0.]
] * 3, device=device)

# 高斯尺度
scales = torch.full(
    (3, 3), 0.15,
    device=device
)

# 不透明度
opacities = torch.full(
    (3,), 0.8,
    device=device
)

# RGB颜色
colors = torch.tensor([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
], device=device)

# 世界坐标系到相机坐标系
viewmats = torch.eye(
    4, device=device
).unsqueeze(0)

# 相机内参
Ks = torch.tensor([[
    [100.,   0., 64.],
    [  0., 100., 64.],
    [  0.,   0.,  1.]
]], device=device)

render, alpha, meta = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=128,
    height=128,
    packed=False,
)

torch.cuda.synchronize()

print("Render shape:", render.shape)
print("Alpha shape:", alpha.shape)
print("Render max:", render.max().item())

assert torch.isfinite(render).all()
assert render.max().item() > 0

# 验证可微分渲染
loss = render.square().mean()
loss.backward()

torch.cuda.synchronize()

assert means.grad is not None
assert torch.isfinite(means.grad).all()
assert means.grad.abs().sum().item() > 0

print("Loss:", loss.item())
print("Gradient of means:")
print(means.grad)

print("Gaussian rendering: PASS")
print("Differentiable rendering: PASS")
print("gsplat test: PASS")
