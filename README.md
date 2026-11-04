# testRepo

Test repository for dsh-superagent.

本项目是一个航迹处理模块的端到端神经网络学习。

## 文档

- [航迹处理端到端神经网络：强化学习可行方案](docs/rl-track-processing-approaches.md)

## 目录结构

```
docs/          方案与设计文档
src/trackrl/   Python 包：运动模型、卡尔曼滤波基线、Gymnasium 仿真环境
tests/         单元测试
```

## 开发环境

需要 Python 3.10 及以上版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # 基础依赖 + 测试工具
pip install -e ".[dev,train]"  # 另外安装 PyTorch 与 Stable-Baselines3，用于训练
pytest
```
