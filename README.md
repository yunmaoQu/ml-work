# 二手车价格预测项目

## 项目简介
本项目基于机器学习方法，构建了多种回归模型（线性回归、Ridge、Lasso、ElasticNet、随机森林、梯度提升、XGBoost）对二手车价格进行预测，并实现了数据可视化和Web在线预测系统。

## 目录结构
```
├── app/                # Streamlit Web应用
│   └── app.py
├── data/               # 数据文件夹
│   ├── 二手车数据.xlsx  # 原始数据
│   ├── processed/      # 预处理后数据
│   └── featured/       # 特征工程后数据
├── models/             # 训练好的模型及编码器
├── results/            # 结果与可视化图表
├── utils/              # 工具函数
├── main.py             # 主程序（数据处理+训练+评估+可视化）
├── visualization.py    # 生成可视化图表
├── requirements.txt    # 依赖包列表
├── README.md           # 使用说明
```

## 环境准备
建议使用Python 3.10+，推荐使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv
# 激活虚拟环境（Windows）
venv\Scripts\activate
# 激活虚拟环境（Linux/Mac）
source venv/bin/activate
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 数据准备
将原始数据文件`二手车数据.xlsx`放在`data/`目录下。

## 一键运行（推荐）
执行主程序，自动完成数据预处理、特征工程、模型训练、评估和可视化：

```bash
python main.py
```

## 演示视频
如需快速了解项目效果，可观看下方演示视频：

[![B站演示视频](https://img.shields.io/badge/Bilibili-演示视频-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1MDEzz4EXL)  
或下载本地视频文件： [点击下载演示.mp4](./show.mp4)

## 生成可视化图表
如需单独生成/刷新可视化图表：
```bash
python visualization.py
```
图表会保存在`results/plots/`目录下。

## 启动Web在线预测系统
使用Streamlit运行Web应用：
```bash
streamlit run app/app.py
```
浏览器访问 http://localhost:8501 按提示输入信息即可预测二手车价格。

## 常见问题
- **特征不匹配报错**：请确保`app.py`输入特征与模型训练时一致，建议用本项目自带的app.py。
- **权限/依赖问题**：请确保已激活虚拟环境并安装所有依赖。
- **模型未生成**：请先运行`python main.py`。

## 参考
- 详细报告见[report.md](https://github.com/yunmaoQu/ml-work/blob/main/report.md)
- PPT大纲见`presentation.md`

---
如有问题欢迎提issue或联系作者。 
