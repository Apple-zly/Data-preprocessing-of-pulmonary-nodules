**肺结节医学图像预处理操作步骤：**
1. 数据读取（包括图像，标签等）
2. 图像数据空间（spacing）归一化
3. 图像获得肺部Mask Lung_Mask
4. 图像提出非肺腔部分（获得肺腔掩模的最大外接矩形）
5. 图像数据窗宽窗位处理（骨头部分给定值）
6. 标签世界坐标系转为图像坐标系
7. 标签减去最大外接矩形即原点坐标。