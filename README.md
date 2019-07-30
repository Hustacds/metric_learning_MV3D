说明：由于第一阶段效果并不理想，因此，对论文进行重新构思，即将rnn改为针对3D shape的metric learning。并对多种方法进行比较。
2. main_metric_learning_20190730.py 为该程序的主入口。主要对比欧式距离，triplet和余弦距离对于多视图的影响。