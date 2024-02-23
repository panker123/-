# 潘俊辰代码
代码运训

```
python run.py --model --dataset cqa --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 16 --epochs=3 --clients_number=3
```

联邦学习解释

利用的FedAvg进行代码聚合,改动部分：run.py的184行
