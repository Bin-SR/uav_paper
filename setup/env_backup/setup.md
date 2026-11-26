# GRaD-Nav / uav_paper 环境搭建与复现手册

基线日期：2026-11-25。目标：在新设备上重建已验证的单卡 CUDA / PyTorch / 3DGS / Nerfstudio 环境，而非重新选择最新版依赖。

> 本文依据实际终端日志及此前安装步骤整理。`env_backup` 八个文件的内容尚未提供给本文作者，因此完整传递依赖版本、源码 commit 以备份文件为准；本文不猜测 commit 哈希。当前完成基础环境安装，尚未验证真实场景、导航训练及所有第三方 CUDA 算子。

## 1. 基线与安装方式总览

| 组件 | 基线版本 | 来源/方式 | 验证状态 |
|---|---|---|---|
| Ubuntu | 24.04.5 LTS x86_64 | 系统 | 已运行 |
| GPU | RTX 5090 32 GB，sm_120 | 单卡运行 | CUDA 运算通过 |
| NVIDIA 驱动 | 580.178.04 | 系统已有 | 已验证，不需重装 |
| Python | 3.10，补丁版本见备份 | Conda | 通过 |
| GCC/G++ | 13.3.0 | apt | 编译通过 |
| CMake | 3.28.3 | apt | 版本检查通过 |
| Ninja | 1.11.1（系统） | apt；另有 pip 包，版本见 freeze | 编译通过 |
| Git | 2.43.0 | apt | clone 通过 |
| FFmpeg | 6.1.1 | apt | 版本检查通过 |
| CUDA Toolkit | 12.8.2-1，nvcc 12.8.93 | NVIDIA apt 源 | sm_120 编译通过 |
| torch | 2.7.1+cu128 | pip 官方 CUDA wheel | GPU 与扩展通过 |
| torchvision | 0.22.1+cu128（约束版本） | pip 官方 CUDA wheel | 需核对 freeze |
| NumPy | 1.26.4（约束版本） | pip | 需核对 freeze |
| setuptools | 75.8.2（此前安装方案） | pip | 以 freeze 为准 |
| wheel、packaging | 见 freeze | pip | 构建工具 |
| tinycudann | 2.0 | Git clone + 本地 CUDA 编译 | 已安装；前后向结果未提供 |
| gsplat | 1.5.3 | pip + 首次运行 JIT CUDA 编译 | 渲染与高斯中心梯度通过 |
| nerfstudio | 1.1.5 | Git clone v1.1.5 + patch + pip editable | 导入与 CLI 通过 |
| nerfacc | 0.5.2 | pip，Nerfstudio 依赖 | 导入通过，CUDA 算子待验证 |
| gym | 0.26.2 | pip | 导入通过 |
| transformers | 4.49.0 | pip | 导入通过 |
| uav_paper | 43637fe（当时 HEAD） | Git clone | 项目运行待验证 |

系统中还曾检测到 Ubuntu 自带 `nvidia-cuda-toolkit` 12.0.140；这是共存组件，不是复现所需版本，不建议在新设备安装它。驱动显示 CUDA 13.0 表示驱动支持的 CUDA 能力，不代表 nvcc 版本。

## 2. 备份文件及恢复原则

建议将本文放入 `~/env_backup/setup.md`，与以下文件一起复制到新设备：

| 文件 | 用途 |
|---|---|
| grad_nav_constraints.txt | pip 关键版本约束，防止 torch/gsplat 被替换 |
| grad_nav_requirements.txt | pip freeze，完整 Python 包版本及可能的本地路径 |
| grad_nav_env.yml | Conda 环境与 pip 依赖记录 |
| grad_nav_explicit.txt | Conda 包精确构建与下载地址（同平台适用） |
| nerfstudio_commit.txt | Nerfstudio 精确源码提交 |
| nerfstudio_patch.diff | Nerfstudio 本地修改 |
| tinycudann_commit.txt | tiny-cuda-nn 精确源码提交 |
| tinycudann_submodules.txt | tiny-cuda-nn 子模块提交记录 |

推荐按本文分阶段重建，再对照 freeze；不要直接把 freeze 全量安装到空环境，因为可能包含旧设备绝对路径、editable 项目、CUDA wheel 和需源码构建的包。`conda create --name grad_nav --file ~/env_backup/grad_nav_explicit.txt` 可作为同 Linux x86_64 平台的 Conda 层恢复替代方案，但不恢复 pip 包、系统 CUDA 或源码。不要与下面的 conda create 重复执行。YAML 也不是跨硬件万能锁文件。

## 3. 系统准备

```bash
cat /etc/os-release
uname -m
nvidia-smi
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config git wget curl ffmpeg libgl1 libglib2.0-0t64
cmake --version
ninja --version
gcc --version
g++ --version
git --version
ffmpeg -version
```

Ubuntu 22.04 等系统的包名可能不同（如 libglib2.0-0）。驱动若已正常工作，不要为了安装 Toolkit 重装驱动。新设备若未安装驱动，先按该系统和 GPU 的官方支持方案安装并验证 nvidia-smi。本文不锁定所有 apt 包的构建号，因为原机器未保存完整 dpkg 清单。

## 4. 创建 Conda 环境

先安装适用于 Linux x86_64 的 Miniconda（已有则跳过），确保 `conda --version` 正常。原机器 base Python 3.14 不适合本基线，必须进入独立环境。

```bash
conda create -n grad_nav --override-channels -c conda-forge python=3.10 pip -y
conda activate grad_nav
which python
python --version
python -m pip --version
python -m pip install --upgrade pip
```

若使用 defaults 遇到 CondaToSNonInteractiveError，可使用上述 conda-forge 方案，或自行阅读并接受 defaults 条款后再创建。不要在创建失败后继续向 base 安装。

## 5. 安装 PyTorch（pip wheel）

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 setuptools==75.8.2 wheel packaging
```

若备份中的 setuptools 等构建工具版本不同，优先使用实际快照。不要照搬旧 Nerfstudio README 的 torch 2.1.2+cu118 / CUDA 11.8，它不是 RTX 5090 的本次适配基线。

### 验收：单卡 PyTorch 运算

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch, torchvision, numpy
print(torch.__version__, torchvision.__version__, numpy.__version__)
print(torch.version.cuda, torch.cuda.get_arch_list())
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
a=torch.randn(4096,4096,device='cuda'); b=a@a
torch.cuda.synchronize()
assert torch.isfinite(b).all()
print('CUDA computation: PASS', b.shape)
PY
```

预期 torch 2.7.1+cu128、CUDA 12.8、RTX 5090 capability (12,0)、架构列表含 sm_120。后续统一用 CUDA_VISIBLE_DEVICES=0，仅使用一张卡。

## 6. 安装 CUDA Toolkit 12.8（NVIDIA apt）

```bash
cd /tmp
wget -O cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
apt-cache policy cuda-toolkit-12-8
sudo apt install cuda-toolkit-12-8
/usr/local/cuda-12.8/bin/nvcc --version
```

原机器候选 12.8.2-1，下载约 3.657 GB。若需精确版本且仍可用，可指定 `cuda-toolkit-12-8=12.8.2-1`，但其传递 apt 依赖仍可能变化。下载曾自动重定向至 NVIDIA .cn 域名并成功。不要安装 `cuda`、`cuda-drivers` 或 Ubuntu 默认 `nvidia-cuda-toolkit` 来替代本步骤。

### Conda 环境变量

```bash
conda activate grad_nav
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda.sh" <<'SH'
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export TCNN_CUDA_ARCHITECTURES=120
SH
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda.sh" <<'SH'
# 移除本环境添加的 CUDA bin，保留 Conda 自身的 PATH 管理
case "$PATH" in /usr/local/cuda-12.8/bin:*) export PATH="${PATH#/usr/local/cuda-12.8/bin:}";; esac
unset CUDA_HOME TORCH_CUDA_ARCH_LIST TCNN_CUDA_ARCHITECTURES
SH
conda deactivate
conda activate grad_nav
which nvcc
nvcc --version
nvcc --list-gpu-code | grep sm_120
python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)'
```

上述脚本假定没有需要保留的其他 CUDA 环境变量；若已有自定义设置应先备份并调整。不要全局设置 LD_LIBRARY_PATH，以免覆盖 PyTorch 自带库。其他 GPU 架构需修改两个架构变量，例如 RTX 30 系列通常为 8.6 / 86（具体以设备能力为准）。

### 验收：PyTorch CUDA 扩展编译

```bash
mkdir -p ~/cuda_test
cd ~/cuda_test
cat > test_torch_cuda.py <<'PY'
import torch
from torch.utils.cpp_extension import load_inline
cpp = 'torch::Tensor add_one_cuda(torch::Tensor x);'
cu = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
__global__ void kernel(const float* x,float* y,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)y[i]=x[i]+1;
}
torch::Tensor add_one_cuda(torch::Tensor x){
 auto y=torch::empty_like(x); int n=x.numel();
 kernel<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>(x.data_ptr<float>(),y.data_ptr<float>(),n);
 C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}
'''
m=load_inline(name='grad_nav_cuda_test',cpp_sources=cpp,cuda_sources=cu,functions=['add_one_cuda'],with_cuda=True,verbose=True)
x=torch.arange(10,device='cuda',dtype=torch.float32)
y=m.add_one_cuda(x); torch.cuda.synchronize()
assert torch.allclose(y,x+1)
print(y); print('CUDA extension test: PASS')
PY
CUDA_VISIBLE_DEVICES=0 MAX_JOBS=4 python test_torch_cuda.py
```

原机器编译日志包含 `-gencode=arch=compute_120,code=sm_120`，成功生成 .so 并加载，输出 1 到 10。缓存位于 ~/.cache/torch_extensions/py310_cu128。换设备应重新编译，不复制旧机器 .so 缓存。

## 7. 保护关键依赖

```bash
mkdir -p ~/env_backup ~/Bin_SR/dependencies
# 若已复制原备份，保留原文件并核对内容；以下为基线内容
cat > ~/env_backup/grad_nav_constraints.txt <<'TXT'
torch==2.7.1+cu128
torchvision==0.22.1+cu128
numpy==1.26.4
gsplat==1.5.3
TXT
export PIP_CONSTRAINT="$HOME/env_backup/grad_nav_constraints.txt"
python -m pip check
```

每个新终端安装依赖前重新 export。约束不会主动安装包，只限制解析版本；出现冲突先检查，不要直接移除约束。

## 8. tiny-cuda-nn（Git + CUDA 源码编译）

```bash
cd ~/Bin_SR/dependencies
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn
git checkout --detach "$(cat ~/env_backup/tinycudann_commit.txt)"
git submodule update --init --recursive
git submodule status --recursive
diff -u ~/env_backup/tinycudann_submodules.txt <(git submodule status --recursive)
CUDA_VISIBLE_DEVICES=0 TCNN_CUDA_ARCHITECTURES=120 MAX_JOBS=8 python -m pip install --no-build-isolation --no-deps -v ./bindings/torch
python -m pip show tinycudann
```

若没有备份 commit，不应假定当前 master 与原环境一致。原机器版本显示 2.0；精确源码以 commit 为准。系统 Ninja 已足够调用，若 Python 依赖要求 ninja 包，pip 会安装对应发行包。

### 验收：HashGrid + MLP 前后向（原机器仅确认安装，建议补测）

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch, tinycudann as tcnn
m=tcnn.NetworkWithInputEncoding(3,4,
 {'otype':'HashGrid','n_levels':8,'n_features_per_level':2,'log2_hashmap_size':15,'base_resolution':16,'per_level_scale':1.5},
 {'otype':'FullyFusedMLP','activation':'ReLU','output_activation':'None','n_neurons':64,'n_hidden_layers':2}).cuda()
x=torch.rand(1024,3,device='cuda',requires_grad=True)
y=m(x); y.float().square().mean().backward(); torch.cuda.synchronize()
assert torch.isfinite(y).all() and x.grad is not None and torch.isfinite(x.grad).all()
print('tiny-cuda-nn test: PASS')
PY
```

## 9. gsplat（pip + JIT 编译）

```bash
python -m pip install gsplat==1.5.3
cd ~/cuda_test
cat > test_gsplat.py <<'PY'
import torch, gsplat
from gsplat import rasterization
d='cuda:0'
means=torch.tensor([[-.5,0.,3.],[0.,0.,3.],[.5,0.,3.]],device=d,requires_grad=True)
quats=torch.tensor([[1.,0.,0.,0.]]*3,device=d)
scales=torch.full((3,3),.15,device=d)
opacities=torch.full((3,),.8,device=d)
colors=torch.eye(3,device=d)
viewmats=torch.eye(4,device=d)[None]
Ks=torch.tensor([[[100.,0.,64.],[0.,100.,64.],[0.,0.,1.]]],device=d)
r,a,_=rasterization(means=means,quats=quats,scales=scales,opacities=opacities,colors=colors,viewmats=viewmats,Ks=Ks,width=128,height=128,packed=False)
assert r.shape==(1,128,128,3) and torch.isfinite(r).all() and r.max()>0
r.square().mean().backward(); torch.cuda.synchronize()
assert means.grad is not None and torch.isfinite(means.grad).all() and means.grad.abs().sum()>0
print(gsplat.__version__,r.shape,a.shape,means.grad)
print('Gaussian rendering: PASS'); print('Differentiable rendering: PASS'); print('gsplat test: PASS')
PY
CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8 python test_gsplat.py
```

原机器首次 JIT 32.98 秒，RGB shape [1,128,128,3]，alpha [1,128,128,1]，render max 0.79563，loss 0.00307829。验证的是高斯中心梯度，不是相机位姿梯度。

## 10. Nerfstudio（Git + patch + editable pip）

原版 v1.1.5 要求 gsplat==1.4.0；本基线保留已通过测试的 1.5.3。修改依赖声明不是证明所有 API 兼容。

```bash
cd ~/Bin_SR/dependencies
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
git checkout --detach "$(cat ~/env_backup/nerfstudio_commit.txt)"
git apply --check ~/env_backup/nerfstudio_patch.diff
git apply ~/env_backup/nerfstudio_patch.diff
git diff -- pyproject.toml
grep -nE 'gsplat|nerfacc|torch|torchvision' pyproject.toml
python -m pip install --dry-run -e .
python -m pip install -e .
python -m pip check
```

无备份时可 checkout v1.1.5，再用 `sed -i 's/"gsplat==1.4.0"/"gsplat==1.5.3"/' pyproject.toml` 并检查 diff；不要与 git apply 重复执行。editable 安装后不要移动源码目录。nerfacc 0.5.2 等依赖由 pip 自动安装。

### 验收：核心导入及 CLI

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch, gsplat, nerfacc, nerfstudio
from importlib.metadata import version
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
for p in ['torch','torchvision','numpy','gsplat','nerfacc','nerfstudio','tinycudann']:
 print(p,version(p))
print('Nerfstudio import test: PASS')
PY
CUDA_VISIBLE_DEVICES=0 ns-train --help
CUDA_VISIBLE_DEVICES=0 ns-render --help
CUDA_VISIBLE_DEVICES=0 ns-viewer --help
```

以上在原机器全部通过。`torch.cuda.amp.custom_fwd/custom_bwd` FutureWarning 属于弃用警告，不必修改。真实 checkpoint 加载与渲染仍待验证。

## 11. GRaD-Nav 额外依赖与项目源码

```bash
python -m pip install gym==0.26.2 transformers==4.49.0
python -m pip check
python -c 'import gym, transformers; print(gym.__version__, transformers.__version__)'
cd ~/Bin_SR
git clone https://github.com/Bin-SR/uav_paper.git
cd uav_paper
git log -1 --oneline
# 如需严格复现当时项目代码：确认该提交存在后执行
# git checkout --detach 43637fe
```

私有仓库需要自己的 GitHub 授权（SSH 或 token），不要将凭据写入本文。原始上游为 https://github.com/Qianzhong-Chen/grad_nav 。Gym 停止维护的警告可暂时忽略，保留 NumPy 1.26.4，不擅自迁移 Gymnasium。项目无 requirements 文件（当时检查深度 4），完整依赖还要以实际代码为准。

```bash
find . -maxdepth 3 -type f -not -path './.git/*' | sort
grep -RnE '^[[:space:]]*(from|import) ' --include='*.py' --exclude-dir=.git .
find . -type f \( -name '*.ply' -o -name '*.ckpt' -o -name '*.pth' -o -name '*.pt' -o -name 'config.yml' -o -name 'transforms.json' \)
```

## 12. 完整 Python 依赖清单与快照对照

以下为日志中明确确认的主要 pip 自动依赖版本；完整清单必须以原始 freeze 为准（本文未读取该文件，不补猜其余版本）。

| 类别 | 已知版本 |
|---|---|
| 渲染与几何 | nerfacc 0.5.2；open3d 0.20.0；pymeshlab 2025.7.post1；trimesh 5.2.2；plyfile 1.1.3；xatlas 0.0.11；Shapely 2.0.7；rtree 1.4.1；fpsample 1.0.2；pyquaternion 0.9.9 |
| 图像与视频 | opencv-python-headless 4.10.0.84；imageio 2.38.0；mediapy 1.2.7；av 17.1.0；rawpy 0.27.1；scikit-image 0.25.2 |
| 科学计算 | scipy 1.15.3；scikit-learn 1.7.2；matplotlib 3.10.9；h5py 3.16.0；tensorly 0.9.0 |
| 深度学习 | timm 0.6.7；torchmetrics 1.9.0；torch-fidelity 0.4.0；pytorch-msssim 1.0.0 |
| 可视化与日志 | viser 0.2.7；tyro 1.0.17；tensorboard 2.20.0；wandb 0.25.0；comet_ml 3.59.0；plotly 7.2.0；jupyterlab 4.6.6 |
| 配置及其他 | protobuf 3.20.3；pyyaml 6.0.4；nuscenes-devkit 1.2.0；yourdfpy 0.0.60；gdown 6.5.2；tqdm 4.70.1 |

查看全部实际版本（包括 torch 的 nvidia-* CUDA runtime wheel、transformers 的传递依赖及 Python Ninja）：

```bash
cat ~/env_backup/grad_nav_requirements.txt
cat ~/env_backup/grad_nav_env.yml
```

### 使用完整备份生成精确版本约束（推荐）

在新设备安装前，可从原 freeze 提取普通 `name==version` 行，保留完整原文件不动。源码包、直接 URL、本地路径应另行处理；不能把它们直接替换成 PyPI 同名包。

```bash
grep -E '^[A-Za-z0-9_.-]+==[^[:space:]]+$' ~/env_backup/grad_nav_requirements.txt \
  | grep -viE '^(nerfstudio|tinycudann|tiny-cuda-nn)==' \
  > ~/env_backup/grad_nav_full_constraints.txt
export PIP_CONSTRAINT="$HOME/env_backup/grad_nav_full_constraints.txt"
```

若希望尽量精确恢复传递依赖，可在第 7 节之后、安装源码包之前使用这个完整约束替代关键约束；若某个版本已不可下载或不支持新平台，记录差异后再有针对性地调整。约束不会自动安装所有包。完成后比较：

```bash
python -m pip freeze > /tmp/grad_nav_new_freeze.txt
diff -u ~/env_backup/grad_nav_requirements.txt /tmp/grad_nav_new_freeze.txt
python -m pip check
```

## 13. 最终验收与故障排查

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import importlib
mods='torch torchvision numpy scipy matplotlib cv2 open3d gym transformers nerfstudio nerfacc tinycudann gsplat yaml tqdm PIL trimesh plyfile tensorboard'.split()
failed=[]
for name in mods:
 try:
  m=importlib.import_module(name); print('[PASS]',name,getattr(m,'__version__',''))
 except Exception as e:
  failed.append(name); print('[FAIL]',name,repr(e))
assert not failed, failed
PY
```

| 现象 | 处理 |
|---|---|
| nvcc not found / 版本 12.0 | 检查 conda 激活脚本、which nvcc、CUDA_HOME |
| no kernel image / unsupported architecture | 检查 sm_120、torch CUDA wheel、第三方扩展编译架构 |
| CUDA version mismatch | torch.version.cuda 与 nvcc 应均为 12.8 |
| 编译 OOM | MAX_JOBS=2 或 4，查看系统 RAM |
| pip ResolutionImpossible | 检查约束和 pyproject，不直接降级 torch/gsplat |
| 旧 .so 加载失败 | 确认源码和 ABI，清理对应扩展缓存后重编译 |
| FutureWarning / Gym 维护提示 | 非致命警告，先保持原环境 |
| cv2.imshow 不可用 | 当前是 headless OpenCV；仅在项目确需 GUI 时处理，避免多个 cv2 wheel 共存 |
| torch.load 权重加载失败 | PyTorch 2.6+ 默认 weights_only=True；仅对可信权重考虑显式 weights_only=False |
| 找不到场景 | 检查资产是否另行下载、配置与硬编码路径 |

当前原机已通过：PyTorch GPU 运算、CUDA 扩展、gsplat 前后向、Nerfstudio 核心导入与 CLI、Gym/Transformers 导入。尚待：tinycudann 实际前后向（未见日志）、nerfacc CUDA、真实 3DGS checkpoint、相机位姿梯度、轨迹规划与训练。

## 14. 后续更新备份

以下命令用于环境验证成功后生成新快照；建议先备份旧目录，不要覆盖唯一可用基线。

```bash
conda activate grad_nav
B="$HOME/env_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$B"
cp ~/env_backup/grad_nav_constraints.txt "$B/"
conda env export --no-builds > "$B/grad_nav_env.yml"
conda list --explicit > "$B/grad_nav_explicit.txt"
python -m pip freeze > "$B/grad_nav_requirements.txt"
git -C ~/Bin_SR/dependencies/nerfstudio rev-parse HEAD > "$B/nerfstudio_commit.txt"
git -C ~/Bin_SR/dependencies/nerfstudio diff > "$B/nerfstudio_patch.diff"
git -C ~/Bin_SR/dependencies/tiny-cuda-nn rev-parse HEAD > "$B/tinycudann_commit.txt"
git -C ~/Bin_SR/dependencies/tiny-cuda-nn submodule status --recursive > "$B/tinycudann_submodules.txt"
# 新增推荐记录
python -m pip check > "$B/pip_check.txt"
dpkg-query -W > "$B/apt_packages.txt"
nvidia-smi > "$B/nvidia_smi.txt"
nvcc --version > "$B/nvcc_version.txt"
git -C ~/Bin_SR/uav_paper rev-parse HEAD > "$B/uav_paper_commit.txt"
git -C ~/Bin_SR/uav_paper diff > "$B/uav_paper_patch.diff"
cp -a "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d" "$B/"
cp ~/cuda_test/test_*.py "$B/"
cp ~/env_backup/setup.md "$B/" # 若本文已放入该目录
```

另行备份未跟踪源码、场景资产、模型权重、配置及实验输出；Git diff 不包含未跟踪文件。分享备份前检查是否含 token、私有下载地址或其他凭据。

## 15. 官方参考

- PyTorch 历史版本：https://pytorch.org/get-started/previous-versions/
- CUDA 12.8 安装：https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/
- tiny-cuda-nn：https://github.com/NVlabs/tiny-cuda-nn
- gsplat：https://github.com/nerfstudio-project/gsplat
- Nerfstudio：https://github.com/nerfstudio-project/nerfstudio
- GRaD-Nav：https://github.com/Qianzhong-Chen/grad_nav
