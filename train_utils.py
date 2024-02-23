import os
import shutil
import logging
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer,CustomLRTrainerCallback

def get_config_dir(args):
    # 构建配置目录的路径，包含各种训练参数信息，用于保存模型检查点和训练日志
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics,w_glob):
    # 设置随机种子
    set_seed(run)

    # 从预训练模型加载T5模型
    model = T5ForConditionalGeneration.from_pretrained("model")
    model.load_state_dict(w_glob)
    #config = AutoConfig.from_pretrained("model", trust_remote_code=True, pre_seq_len=128)
    #model = AutoModel.from_pretrained("model", config=config, trust_remote_code=True)
    # 如果启用了并行化，则使用 model.parallelize() 进行并行化处理
    if args.parallelize:
        model.parallelize()

    # 根据训练参数构建配置目录的路径
    config_dir = get_config_dir(args)
    # 模型检查点保存路径
    output_dir = f'ckpts/{config_dir}/{run}'  # 用于保存模型检查点
    # 训练日志保存路径
    logging_dir = f'logs/{config_dir}/{run}'  # 用于保存训练日志

    # 如果设置了不记录日志，则选择不记录
    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # 如果模型检查点路径已存在，则清空该路径
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir,  # 输出目录，保存模型检查点等文件
        optim='adamw_torch',
        remove_unused_columns=False,  # 是否移除未使用的列
        evaluation_strategy='no',  # 评估策略，按步数进行评估
        save_strategy='no',  # 保存策略，设置为不保存
        logging_dir=logging_dir,  # 训练日志保存路径
        logging_strategy=logging_strategy,  # 日志记录策略
        logging_steps=args.eval_steps,  # 每隔多少步记录一次日志
        #logging_steps=2,  # 每隔多少步记录一次日志
        max_steps=args.max_steps,  # 最大训练步数
        learning_rate=args.lr,  # 学习率
        lr_scheduler_type='constant',
        gradient_accumulation_steps=args.grad_steps,  # 梯度累积步数
        per_device_train_batch_size=args.batch_size,  # 每个设备上的训练批量大小
        predict_with_generate=True,  # 是否使用生成模式进行预测
        seed=run,  # 随机种子
        local_rank=args.local_rank,  # 本地排名
        bf16=args.bf16,  # 是否使用 bf16 数据类型
        generation_max_length=args.gen_max_len,  # 生成序列的最大长度
        prediction_loss_only=False,  # 是否只计算预测损失
    )

    if args.model_type == 'task_prefix':
        # 如果模型类型为 'task_prefix'，则使用 TaskPrefixDataCollator 创建 data_collator 对象
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        # 如果模型类型为 'standard'，则使用 DataCollatorForSeq2Seq 创建 data_collator 对象
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        # 如果模型类型不是上述两种之一，抛出 ValueError
        raise ValueError

    # 构建 Seq2SeqTrainer 初始化参数的字典
    trainer_kwargs = {
        'alpha': args.alpha,  # 混合参数
        'model': model,  # 模型
        'output_rationle':True,
        'args': training_args,  # 训练参数
        'train_dataset': tokenized_datasets,  # 训练数据集
        #'eval_dataset': {'test': tokenized_datasets['valid'] },  # 评估数据集
        'data_collator': data_collator,  # 数据收集器
        'tokenizer': tokenizer,  # 分词器
        #'callbacks' :[CustomLRTrainerCallback()],
        'compute_metrics': compute_metrics,  # 评估指标计算函数
    }

    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError

    trainer.train()
    return trainer.model.state_dict()

def validate_model(args, run, tokenizer, tokenized_datasets, compute_metrics,w_glob):
    model = T5ForConditionalGeneration.from_pretrained("model")
    model.load_state_dict(w_glob)
    # 定义训练参数，确保设置相关参数，例如 output_dir, per_device_eval_batch_size 等
    training_args = Seq2SeqTrainingArguments(
        remove_unused_columns=False,  # 是否移除未使用的列
        output_dir="./validation_output",
        per_device_eval_batch_size=args.batch_size,  # 适当设置评估时的 batch size
        logging_dir="./validation_logs",
        gradient_accumulation_steps=args.grad_steps,
        evaluation_strategy="steps",  # 设置为 "steps" 以便控制评估的频率
        eval_steps=100,  # 评估的步数，可以根据需要调整
        seed=run,  # 随机种子
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
        predict_with_generate = True  # 是否使用生成模式进行预测
    )

    if args.model_type == 'task_prefix':
        # 如果模型类型为 'task_prefix'，则使用 TaskPrefixDataCollator 创建 data_collator 对象
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        # 如果模型类型为 'standard'，则使用 DataCollatorForSeq2Seq 创建 data_collator 对象
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        # 如果模型类型不是上述两种之一，抛出 ValueError
        raise ValueError

    # 创建 Seq2SeqTrainer
    trainer_kwargs = {
        'alpha': args.alpha,  # 混合参数
        'model': model,  # 模型
        'output_rationle': True,
        'args': training_args,  # 训练参数
        'train_dataset': None,  # 训练数据集
        'eval_dataset': tokenized_datasets,  # 评估数据集
        'data_collator': data_collator,  # 数据收集器
        'tokenizer': tokenizer,  # 分词器
        'compute_metrics': compute_metrics,  # 评估指标计算函数
    }

    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError

    # 执行验证
    with torch.no_grad():
        results = trainer.evaluate()

    # 打印评估结果
    print(results)