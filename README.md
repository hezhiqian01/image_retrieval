# image_retrieval

# 0.简介
对论文《Scalable Recognition with a Vocabulary Tree》的实现
# 1. 代码结构
- models-----------------------------------# 默认的模型保存文件
- prediction_result------------------------# 默认的预测结果文件目录
- train_images-----------------------------# 默认的训练数据库图片文件目录
- logs-------------------------------------# 运行日志文件目录
- src--------------------------------------# 项目源码
- requirements-----------------------------# python依赖包
- ReadMe.md

# 2. 环境配置
```bash
pip install -r requirements
```

# 3. 训练流程
```bash
# 训练模型 
# k指字典树的子节点数量，l指字典树的深度
python -m src.train --image_dir train_images --model_output models -k 10 -l 5
```


# 4. 测试流程
```bash
python src/predict.py --model_path models --images image_path --output prediction_result
```

