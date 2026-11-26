# Standford-uav GRaD-Nav

## 1.预备知识  

### 1.1 渲染 
**NeRF 和 3DGS**  
#### NeRF   
全称是 Neural Radiance Fields，中文叫神经辐射场。  
它的核心思想是：

利用神经网络学习三维空间中的位置与观察方向到颜色和体密度的映射。  
$$F_\theta(\mathbf{x},\mathbf{d}) \rightarrow (\mathbf{c},\sigma)$$
其中：
- $(\mathbf{x}=(x,y,z))$：三维空间位置；
- $(\mathbf{d})$：观察方向；
- $(\mathbf{c}=(r,g,b))$：颜色；
- $(\sigma)$：体密度；
- $(\theta)$：神经网络参数。
训练完成后，可以通过体渲染（Volume Rendering）得到某个相机视角下的图像。  

NeRF 的意义在于：不需要传统意义上显式构建完整的三角网格，也可以学习一个能够进行新视角渲染的三维场景表示。  

#### 3D Gaussian Splatting（3DGS）  
与 NeRF 都可以用于三维场景表示，但实现方式不同。  
NeRF 主要通过神经网络隐式表示场景  
3DGS 则通过大量具有空间位置、尺度、旋转、透明度和颜色属性的三维高斯分布来显式表示场景  
单个高斯可以表示为：
$G(\mathbf{x})= \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$  

大量高斯组合起来，就能表示复杂的三维场景  

Nerfstudio仓库，它主要承担三维场景表示与相关计算的基础设施角色  
gsplat：提供高斯泼溅相关的高效计算能力  
GRaD-Nav：在这些基础上构建无人机视觉导航和强化学习系统    

### 1.2 GRaD-Nav  
#### 要解决的问题：  
传统无人机通常的步骤：感知周围环境，获取障碍物与目标位置，路径规划，生成轨迹，控制无人机跟踪轨迹  
GRad-Nav: 希望无人机可以直接根据**视觉信息**学习导航策略，而不是完全以来传统的显式地图、路径规划及轨迹控制流  
#### 用到的技术  
3D Gaussian Splatting(3DGS), Differentiable Dynamics(可微分动力学), RL强化学习  
#### 核心思想  
技术路线可以理解成：3DGS场景->视觉观测->导航策略->无人机控制  
关键设计：**把可微分的场景渲染和无人机动力学，纳入强化学习的优化过程**  
传统的强化学习训练中，策略网络输出actions后，无人机在仿真环境中运动，获得下一时刻的obs，但是环境中的渲染和动力学通常被当作不可微分的黑盒  
GRaD-Nav则利用可微分模块，让**梯度**能够通过视觉观测和动力学过程传播，为策略优化提高更丰富的梯度信息  
#### 系统架构        
- 环境表示-3DGS  
- 无人机动力学-Differentiable Dynamics  
一般来说，四旋翼的状态可以写成: $s_t = [p_t, v_t, q_t, \omega_t]$  
其中，$p_t$表示位置，$v_t$表示线速度，$q_t$表示姿态四元数，$\omega_t$表示角速度  
动作$a_t$可以是期望推力和例句，也可以是其他控制量，具体取决于代码中的动力学接口  
动力学表达式：$s_{t+1}=f(s_t,a_t)$，并且还要求可微，即
$$
\frac{\partial s_{t+1}}{\partial s_t}
,\qquad
\frac{\partial s_{t+1}}{\partial a_t}
$$

- 视觉策略网络-Actor-Critic
- CENet-Context aided Estimator Network  
CENet 的作用是帮助策略估计环境变化及相关上下文信息,目的就是帮助策略更好地适应环境变化。
#### 关于坐标系的问题
Nerfstudio 导出的场景存在坐标归一化问题，不能直接把原始坐标当作无人机导航坐标。  
在CUSTOM_SCENE.md中，在GPT的第一次问答中的3.4




## 2.复现工作  
GRaD-Nav 所研究的是另一种视觉导航策略学习路线  
先根据/env_backup/setup.md, 配置环境  

项目总览：  
```
真实场景数据
       ↓
3D Gaussian Splatting 场景表示
       ↓
根据无人机位姿渲染视觉观测
       ↓
神经网络导航策略
       ↓
控制动作
       ↓
可微分无人机动力学
       ↓
更新无人机状态
       ↓
获得新的视觉观测
```  
核心模块  
| 核心模块 | 主要作用 |
|---|---|
| 3D Gaussian Splatting | 表示真实场景并生成视觉观测 |
| Differentiable Dynamics | 建立支持梯度传播的无人机动力学 |
| Navigation Policy | 学习从观测到控制动作的映射 |
| Trajectory Optimization | 根据导航目标优化运动轨迹或策略 |  

GRaD-Nav++ 则在此基础上进一步引入视觉语言模型，实现更加复杂的语义导航任务  
初步规划：  
| 实验 | 实验内容 | 最终验收标准 |
|---|---|---|
| 实验1 | 项目结构、数据及运行环境检查 | 训练入口能够正常启动 |
| 实验2 | 点云地图加载与可视化 | 能够独立显示三维场景 |
| 实验3 | 3DGS 场景渲染与虚拟相机 | 给定相机位姿，能够生成 RGB 图像 |
| 实验4 | 无人机可微分动力学 | 能够独立控制无人机运动，并验证梯度传播 |
| 实验5 | Navigation Env 导航环境 | 能够完成 `reset()`、`step()` 和观测获取 |
| 实验6 | GRaD-Nav 策略网络与训练算法 | 理解完整训练过程并运行短时间训练 |
| 实验7 | GRaD-Nav 完整复现 | 完成 Long Trajectory、Multi-Gate 训练和测试 |
| 实验8 | PPO、BPTT 基线算法 | 对比不同算法的训练效率与导航效果 |
| 实验9 | GRaD-Nav++ | 完成 VLM、VLA、MoE 相关实验 |
| 实验10 | 自定义场景与算法改进 | 在自己的场景中训练导航策略 |  

### 2.1 项目结构与总览  
从README.md中，Training GRaD-Nav:  
`python examples/train_gradnav.py --cfg examples/cfg/gradnav/drone_long_traj.yaml --logdir examples/logs/DroneLongTraj/gradnav`  
由`train_gradnav.py`向下寻找调用关系  

三个核心理论：  
Gaussian Splatting，3DGS使用大量三维高斯表示场景  
Differentiable Dynamics，可微分动力学，让策略训练能够利用系统动力学的梯度信息  
与PPO的区别：  
| 对比项目 | PPO | 基于可微分动力学的训练 |
|---|---|---|
| 是否需要动力学梯度 | 不需要 | 需要利用 |
| 优化依据 | 奖励、优势函数 | 可微分的任务损失等 |
| 环境能否为黑盒 | 可以 | 梯度传播部分不能完全是黑盒 |
| 策略更新方式 | Policy Gradient | 可以通过动力学反向传播 |
| 典型问题 | 采样效率 | 长时域梯度稳定性、动力学误差 |  
在仓库中加入 PPO 和 BPTT(backpropagation through time, 时序反向传播) 作为基线算法。后续我们不仅要知道哪个算法训练得快，还要理解它为什么快。  

`drone_long_traj.yaml`配置文件  

### 2.2 实验一  
第一层：训练入口，examples/train_gradnav.py  
第二层：算法环境，algorithms/gradnav.py  
第三层：环境加载3DGS场景，envs/drone_long_traj.py，其中的`get_gs()`来自utils/gs_local.py 
第四层：3DGS环境封装，在`get_gs()`中有参数maps，其中的值来自deone_long_traj.yaml配置文件中的 map_name: gate_mid 


因为设备是50系列的显卡，所以没有按照官网要求的torch2.1.2+cu118, 而是装的torch2.7.1+cu128, cuda-toolkit也从11.8变为12.8，所以真正运行时会遇到一些版本的问题或者旧版与新版写法不一样的问题  
eg.torch在2.6之前，torch.load的参数`weights_only`默认是`Flase`, 从2.6开始，默认是`True`  
报错:
```
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy.core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.
```
临时解决：`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 `  
之后再修改源码： 
```
  File "/home/nnn/Bin_SR/dependencies/nerfstudio/nerfstudio/utils/eval_utils.py", line 62, in eval_load_checkpoint
    loaded_state = torch.load(load_path, map_location="cpu")
```  
train_gradnav.py为训练入口，重点在GRaDNav这个类：  
```
traj_optimizer = gradnav.GradNav(cfg_train)  
algorithms/gradnav.py
    └── class GradNav
          ├── __init__()
          ├── train()
          └── play()
```  

第一次训练的日志打印：  
```
(grad_nav) nnn@nnn-System-Product-Name:~/Bin_SR/uav_paper$ TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python examples/train_gradnav.py \
    --cfg examples/cfg/gradnav/drone_long_traj.yaml \
    --logdir examples/logs/DroneLongTraj/gradnav
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
/home/nnn/Bin_SR/dependencies/nerfstudio/nerfstudio/field_components/activations.py:32: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @custom_fwd(cast_inputs=torch.float32)
/home/nnn/Bin_SR/dependencies/nerfstudio/nerfstudio/field_components/activations.py:39: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, g):
Setting seed: 0
device: cuda:0
/home/nnn/miniconda3/envs/grad_nav/lib/python3.10/site-packages/gym/spaces/box.py:127: UserWarning: WARN: Box bound precision lowered by casting to float32
  logger.warn(f"Box bound precision lowered by casting to {self.dtype}")
device:  cuda:0
/home/nnn/Bin_SR/uav_paper/temp_config_resolved.yaml
/home/nnn/miniconda3/envs/grad_nav/lib/python3.10/site-packages/torchmetrics/functional/image/lpips.py:332: UserWarning: Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the`weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False.
  self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
Loading latest checkpoint from load_dir
/home/nnn/Bin_SR/dependencies/nerfstudio/nerfstudio/utils/eval_utils.py:62: UserWarning: Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the`weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False.
  loaded_state = torch.load(load_path, map_location="cpu")
✅ Done loading checkpoint from 
/home/nnn/Bin_SR/uav_paper/envs/assets/gs_data/sv_1007_gate_mid/splatfacto/2024-10-07_145741/nerfstudio_models/step-0000
29999.ckpt
/home/nnn/miniconda3/envs/grad_nav/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/nnn/miniconda3/envs/grad_nav/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=SqueezeNet1_0_Weights.IMAGENET1K_V1`. You can also use `weights=SqueezeNet1_0_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth" to /home/nnn/.cache/torch/hub/checkpoints/squeezenet1_0-b66bff10.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.78M/4.78M [00:21<00:00, 232kB/s]
num_envs =  128
num_actions =  4
num_obs =  57
Sequential(
  (0): Linear(in_features=57, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (3): Linear(in_features=512, out_features=256, bias=True)
  (4): ELU(alpha=1.0)
  (5): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  (6): Linear(in_features=256, out_features=128, bias=True)
  (7): ELU(alpha=1.0)
  (8): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (9): Linear(in_features=128, out_features=4, bias=True)
  (10): Identity()
)
Parameter containing:
tensor([-1., -1., -1., -1.], device='cuda:0', requires_grad=True)
Sequential(
  (0): Linear(in_features=67, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (3): Linear(in_features=512, out_features=256, bias=True)
  (4): ELU(alpha=1.0)
  (5): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  (6): Linear(in_features=256, out_features=128, bias=True)
  (7): ELU(alpha=1.0)
  (8): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (9): Linear(in_features=128, out_features=1, bias=True)
)
/home/nnn/Bin_SR/uav_paper/envs/assets/quadrotor_dynamics_advanced.py:132: UserWarning: Using torch.cross without specifying the dim arg is deprecated.
Please either pass the dim explicitly or simply use torch.linalg.cross.
The default value of dim will change to agree with that of linalg.cross in a future release. (Triggered internally at /pytorch/aten/src/ATen/native/Cross.cpp:62.)
  omega_cross = torch.cross(angular_velocity.clone(), torch.bmm(self.I.clone().detach(), angular_velocity.clone().unsqueeze(2)).squeeze(2))
 save best policy with loss 358.46 50
iter 1: ep loss 358.46, ep discounted loss 327.60, vae loss 6.06, ep len 15.4, fps total 201.61, value loss 25754.34, grad norm before clip 215.29, grad norm after clip 1.00
iter 2: ep loss 372.73, ep discounted loss 341.13, vae loss 2.17, ep len 15.1, fps total 180.50, value loss 25880.79, grad norm before clip 118.40, grad norm after clip 1.00
 save best policy with loss 332.53 81
```  
#### 2.2.1 分析日志信息：  
##### 初始化信息
**1.加载3DGS场景**
```
Loading latest checkpoint from load_dir
Done loading checkpoint from /gs_data/sv_1007_gate_mid/splatfacto/2024-10-07_145741/nerfstudio_models/step-000029999.ckpt
```  
使用的场景是**sv_1007_gate_mid**  
场景重建方法是 Nerfstudio 中的**splatfacto**  
加载的模型**step-000029999.ckpt**  
我们现在是在一个已经重建好的场景中训练无人机，这里的checkpoint是已经训练好的3DGS场景模型，因此当前我们不需要重新进行3DGS场景重建

**并行环境数量与观测空间**  
```
num_envs = 128
num_actions = 4
num_obs = 57
```  
| 参数 | 数值 | 含义 |
|---|---:|---|
| `num_envs` | 128 | 并行无人机环境数量 |
| `num_actions` | 4 | 单架无人机的动作维度 |
| `num_obs` | 57 | Actor 网络的输入维度 |  

观测$O_t$，是128×57维，动作$A_t$是128×4维

**Actor网络结构**  
在GRaD-Nav的系统架构图中的：Actor Net 512×256×128
```
57维输入
    ↓
Linear(57, 512)
    ↓
ELU + LayerNorm
    ↓
Linear(512, 256)
    ↓
ELU + LayerNorm
    ↓
Linear(256, 128)
    ↓
ELU + LayerNorm
    ↓
Linear(128, 4)
    ↓
4维动作相关输出
```  
GRaD-Nav 的 Actor 不是直接接收 RGB 图像，而是接收经过编码的低维特征  

Actor的四维输出：在`algorithms/gradnav.py`中，四个值r, p, y, thrust对应四个LFP：    
```
self.train_r_filter
self.train_p_filter
self.train_y_filter
self.train_thrust_filter
```  
四维输出就是roll、pitch、yaw 和 thrust, 输出也做了归一化处理：  
```
actions = self.actor(obs, deterministic=deterministic)
actions = torch.tanh(actions)
```


**Critic 网络结构**  
在GRaD-Nav的系统架构图中的：Critic Net 512×256×128
```
67维输入
    ↓
512
    ↓
256
    ↓
128
    ↓
1维价值估计
```  
**非对称 Actor-Critic：**  
注意到，Actor Net的输入是57维，也就是观测num_obs的维度，但是Critic Net的输入是67维度，多了10  

训练时，Critic 可以利用仿真环境提供的、真实机器人不容易直接获取的信息来辅助策略学习。

例如，真实环境中的无人机未必能直接获取障碍物的精确位置、真实距离或完整环境状态，而仿真环境可以提供这些信息。

Actor 则尽量只使用真实机器人部署时能够获取的观测。

至于额外的10维究竟是什么，需要查看环境中 Actor observation 和 Critic observation 的拼接代码，暂时不能确定。

这种思想对于你以后进行无人机 Sim2Real 非常重要  


**57与67的原因**  
在`envs/drone_long_traj.py`中规定了obs的维度，函数：calculateObservations()  
Actor 的维度57=17+16+24  
Critic的维度67=27+16+24  

| 符号 | 维度 | 含义 |
|---|---:|---|
| $o_t$ | 17 | 当前无人机观测 |
| $e_t$ | 16 | CNN提取的视觉特征 |
| $z_t$ | 24 | CENet提取的潜在特征 |
| 合计 | **57** | Actor完整输入 |  
**注意** z和e的维度可能写反了，原文中z是16维
paper中的策略：$π_θ(a_(t+1)|o_t, z_t, e_t)$

Actor输入的17维观测：对应源码`envs/drone_long_traj.py`中的self.obs_buf  
| 数据 | 维度 | 索引 |
|---|---:|---|
| 无人机高度 | 1 | 0 |
| 垂直速度 | 1 | 1 |
| 姿态四元数 | 4 | 2～5 |
| 当前动作记录 | 4 | 6～9 |
| 上一次动作记录 | 4 | 10～13 |
| 三轴线速度 | 3 | 14～16 |
| 视觉特征 | 16 | 17～32 |
| 潜在特征 | 24 | 33～56 |
| **合计** | **57** | |  

Critic输入的27维观测：对应源码`envs/drone_long_traj.py`中的self.privilege_obs_buf  
| 数据 | 维度 | 索引 |
|---|---:|---|
| 三维位置 | 3 | 0～2 |
| 三轴线速度 | 3 | 3～5 |
| 三轴线加速度 | 3 | 6～8 |
| 姿态四元数 | 4 | 9～12 |
| 三轴角速度 | 3 | 13～15 |
| 机体向上向量的一个分量 | 1 | 16 |
| 航向与运动方向对齐程度 | 1 | 17 |
| 当前动作 | 4 | 18～21 |
| 上一次动作 | 4 | 22～25 |
| 深度相关信息 | 1 | 26 |
| 视觉特征 | 16 | 27～42 |
| 潜在特征 | 24 | 43～66 |
| **合计** | **67** | |


##### 训练日志信息  
以iter 1为例：  
```
iter 1:
ep loss 358.46,
ep discounted loss 327.60,
vae loss 6.06,
ep len 15.4,
fps total 201.61,
value loss 25754.34,
grad norm before clip 215.29,
grad norm after clip 1.00
```  
| 指标 | 中文含义 | 主要反映 |
|---|---|---|
| `iter` | 训练迭代次数 | 训练进度 |
| `ep loss` | Episode累计损失 | 整体导航表现 |
| `ep discounted loss` | 折扣累计损失 | 考虑时间折扣的导航表现 |
| `vae loss` | VAE损失 | 特征编码相关训练效果 |
| `ep len` | 平均Episode长度 | 无人机持续运行步数 |
| `fps total` | 整体训练吞吐率 | 仿真与训练效率 |
| `value loss` | 价值网络损失 | Critic预测误差 |
| `grad norm before clip` | 裁剪前梯度范数 | 原始梯度大小 |
| `grad norm after clip` | 裁剪后梯度范数 | 梯度裁剪效果 |  


#### 2.2.2 系统框架中的CNN：
在训练中的下载了两个预训练模型：AlexNet和SqueezeNet，还有一个VAE编码器(架构图中的CENet)  
| 对比 | AlexNet | SqueezeNet |
|---|---|---|
| 权重文件 | `alexnet-owt-7be5be79.pth` | `squeezenet1_0-b66bff10.pth` |
| 文件大小 | 233 MB | 4.78 MB |
| 所属模块 | Nerfstudio的LPIPS相关模块 | GRaD-Nav视觉感知模块 |
| 主要作用 | 感知图像相似度评价 | 提取导航视觉特征 |
| 对应架构图 | 3DGS框内部的附属功能，未单独画出 | Pretrained CNN |
| 是否直接生成Actor视觉输入 | 否 | 是 |  

##### AlexNet  
```
初始化 Nerfstudio
        ↓
下载 alexnet-owt-7be5be79.pth
        ↓
Loading latest checkpoint from load_dir
        ↓
3DGS checkpoint 加载成功
        ↓
下载 squeezenet1_0-b66bff10.pth
```  
在日志中有`torchmetrics/functional/image/lpips.py`, AlexNet 的加载与 Nerfstudio 中的 LPIPS 图像感知评价模块有关  
LPIPIS：Learned Perceptual Image Patch Similarity，即学习式感知图像块相似度  
LPIPS 越小，通常代表两幅图像在感知特征空间中越接近  

##### SqueezeNet  
该Net对应的是系统架构图中的Pretrained CNN，并且在仓库中有`/model/squeeze_net.py`  
```
3D-GS Model
     ↓ RGB
Pretrained CNN
     ↓ e_t
CENet / Actor Net
```  
它主要是提取图像汇总的特征，并且作者没有直接使用 SqueezeNet 的1000类别分类输出，  
而是保留其卷积特征提取部分，然后添加全局平均池化和线性映射层  
SqueezeNet的参数不参与训练，主干网络是冻结的，不参与梯度更新  

##### CENet  
是一个VAE编码器，将历史观测转换成**24维(?有疑问)**潜在向量, CENet的输出是$z_t$, 并且原文中说明：  
`where zt ∈ R16 is the latent embedding encoded by CENet`  
`algorithms/gradnav.py`中有函数：`compute_vae_loss()`,输入是obs和history_obs

#### 2.2.3 可微分动力学与 Critic 共同优化 Actor  
如何共同优化？  
配置yaml中的参数，steps_num: 32  
每次仅在一个较短的时间范围内(32步)展开动力学，再使用 Critic 估计未来的累计价值  

#### 2.2.4 运行阶段：如何真正渲染RGB图像