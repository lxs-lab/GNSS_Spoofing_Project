GNSS_Spoofing_Project/
│
├── data/
│   ├── raw/                 # [存放原始数据] 把 ds0.bin 到 ds6.bin 全部放这里
│   └── processed/           # [存放生成的CSV] 脚本会自动生成 CSV 放这里
│       └── dataset/         # [存放生成的PT] 最终的图神经网络 .pt 文件放这里
│
├── logs/                    # [日志] 存放训练过程的 loss 曲线图、日志文件
├── models/                  # [模型存档] 存放训练好的 .pth 权重文件
│
├── src/                     # [核心代码库]
│   ├── __init__.py          # 空文件，标识这是一个包
│   ├── config.py            # [控制台] 全局参数配置 (最重要！)
│   ├── extractor.py         # [ETL] 特征提取逻辑 (封装好的类)
│   ├── graph_builder.py     # [图构建] CSV -> PyG Data 逻辑
│   ├── model.py             # [网络] Graph Transformer 模型定义
│   └── utils.py             # [工具] 画图、评价指标计算等
│
├── 1_batch_extract.py       # [主程序] 第一步：批量提取所有 bin 文件的特征
├── 2_build_dataset.py       # [主程序] 第二步：构建训练集和测试集
├── 3_train_eval.py          # [主程序] 第三步：训练与全场景评估
└── README.md                # 项目说明