# 多源异构数据编织Demo
## 项目简介
**本项目是一个面向多源异构数据的数据编织（Data Fabric）Demo。该演示系统通过扫描多源异构数据（包括 RGB 卫星图像、灰度卫星图像、SAR 图像以及军事目标图像），生成统一索引并提取统一的特征向量，将来自不同传感器、不同分辨率和不同格式的数据映射到同一特征空间；随后通过无监督聚类对这些特征进行分组，再利用 UMAP 降维进行可视化，同时在网页端提供簇内随机图像的 4x4 网格展示和交互操作，使用户能够直观探索簇间及簇内的图像相似性和潜在关联，从而实现多源异构数据的整合、融合与可交互分析，以此种方式体现了数据编织（data fabric）的核心理念。**
![image](https://github.com/qwerqwerqwe8688-jpg/-Demo/blob/master/demo1.png)

## 系统架构
### 数据编织层
统一索引构建: 跨数据源的标准化元数据管理

多模态适配: RGB、灰度、SAR、遥感图像的统一处理接口

格式无关性: 支持 PNG、TIFF、JPEG 等多种图像格式

### 特征工程层
深度特征提取: 基于 ResNet50 的通用视觉特征表示

通道自适应: 自动处理单通道、多通道图像数据

特征归一化: 跨数据源的特征空间对齐

### 分析计算层
无监督聚类: K-means 自动聚类分析

聚类数优化: 基于轮廓系数的自适应参数选择

相似性度量: 高维特征空间的相似性计算

### 可视化交互层
降维可视化: UMAP 多维数据投影

聚类探索: 交互式簇内样本浏览

结果导出: 聚类标签和可视化结果持久化

## 支持的数据源
### 遥感对地观测数据
DOTA 数据集: 包含 Google Earth RGB 图像和 GF-2/JL-1 卫星全色波段数据

图像类别: 机场、停车场、码头、公共交通枢纽等基础设施

### 合成孔径雷达数据
OpenSARShip: 船舶目标 SAR 图像

OpenSARUrban: 城市场景 SAR 图像块，涵盖 10 个目标区域类别

### 军事目标识别数据
军事资产数据集: 12 类军事目标，26,315 个标注样本

adomvi 军用车辆: 四类主要军用车辆的细粒度分类

UAV-tracking-tank: 坦克目标的无人机视角图像

## 系统要求
### 软件环境
Python 3.8+

PyTorch 1.8+

scikit-learn 1.0+

### 硬件建议
内存: 16GB+ (取决于数据规模)

存储: 100GB+ 可用空间

GPU: 可选，用于加速特征提取

## 安装部署
### 环境配置
#### 创建虚拟环境
conda create -n datafabric python==3.9
conda activate datafabric

#### 安装依赖包
pip install -r requirements.txt

### 使用流程
阶段一：数据编织准备
#### 构建统一数据索引
python data_fabric.py --root datasets --out fabric_index.csv #--root为你的数据集目录
输出：标准化元数据索引文件

阶段二：特征统一提取
#### 提取深度视觉特征
python feature_extractor.py --index fabric_index.csv --out fabric_features.npy
输出：高维特征向量集合

阶段三：聚类分析
#### 自动聚类分析
python train_classifier.py --index fabric_index.csv --feats fabric_features.npy --model out_clf.joblib
输出：聚类模型和样本标签

阶段四：UMAP生成
#### 本地生成图片
python cluster_visualize.py --index fabric_index.csv --feats fabric_features.npy
输出：本地图片umap.png

阶段四：结果探索
#### 启动交互式分析界面
python app_gradio.py --index fabric_index.csv --feats fabric_features.npy --model out_clf.joblib
访问: http://127.0.0.1:7860

## 核心算法
### 特征提取
骨干网络: ResNet50 (ImageNet 预训练)

特征维度: 2048 维

预处理: ImageNet 标准归一化

### 聚类优化
基础算法: K-means

参数选择: 轮廓系数最大化

搜索范围: K ∈ [2, min(10, n_samples)]

### 可视化方法
降维算法: UMAP (Uniform Manifold Approximation and Projection)

邻居参数: n_neighbors=15

最小距离: min_dist=0.1

## 输出结果
数据产品
fabric_index.csv - 统一数据索引

fabric_features.npy - 特征向量库

cluster_model.joblib - 聚类模型

fabric_index_clustered.csv - 带聚类标签的索引

## 技术特性
### 数据编织能力
异构数据集成: 统一处理多源、多模态图像数据

元数据标准化: 跨数据集的统一元数据模型

质量控制: 自动异常检测和数据验证

### 算法鲁棒性
通道自适应: 智能处理不同通道数的图像数据

格式兼容性: 支持专业图像格式（如多波段 TIFF）

容错处理: 对损坏数据的自动跳过和日志记录

### 可扩展性
模块化设计: 各处理阶段独立可替换

算法插拔: 支持特征提取和聚类算法的灵活更换

接口标准化: 统一的输入输出格式

## 应用场景
跨域数据关联分析

多源遥感数据的自动模式发现

SAR 与光学图像的协同分析

军事目标的多视角关联

重复样本检测

数据分布均匀性分析

异常样本识别

无监督类别发现

数据内在结构探索

特征空间关系分析

## 注意事项
### 数据要求
图像文件应可正常读取

建议数据规模 > 1000 样本以获得有意义的聚类结果

混合数据源时注意分辨率和比例的差异

### 性能考虑
特征提取阶段计算密集，建议批量处理

大规模数据聚类需要足够内存

可视化界面响应时间与数据规模相关

### 结果解释
聚类结果为数据驱动，需结合领域知识解释

UMAP 投影保持拓扑结构但不保证距离意义

自动选择的 K 值需通过领域验证



Fabric 系统通过数据编织技术，为多源异构数据分析提供了演示的解决方案，支持从数据集成到知识发现的完整工作流。