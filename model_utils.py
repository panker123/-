import pandas as pd
import torch
from typing import Any,Dict,List,Optional,Tuple,Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers.trainer_callback import TrainerCallback
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features,return_tensors=None):
        features_df=pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids',
                     'expl_attention_mask': 'attention_mask'}).to_dict('records')

        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            'pred': pred_features,
            'expl': expl_features,
        }

class CustomLRTrainerCallback(TrainerCallback):
    """自定义回调来在每步训练后调整学习率"""
    def on_step_end(self, args, state, control, **kwargs):
        # 每步训练后调用
        optimizer = kwargs.get("optimizer")

        for param_group in optimizer.param_groups:
            # 减少指定的学习率
            param_group['initial_lr'] = max(param_group['initial_lr'] - (5e-5 / (3*args.max_steps)), 0)  # 防止学习率变成负数
            param_group['lr'] = param_group['initial_lr']  # 防止学习率变成负数

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 获取当前优化器和学习率
        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            # 假设我们只有一个参数组
            current_lr = optimizer.param_groups[0]['initial_lr']
            # 将当前学习率添加到日志中
            if logs is not None:  # 确保logs字典存在
                logs["learning_rate"] = current_lr
            #print(f"Current Learning Rate: {current_lr}")



class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self,alpha,output_rationle,**kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.output_rationle=output_rationle

    def compute_loss(self,model,inputs,return_outputs=False):
        inputs
        pred_outputs=model(**inputs['pred'])
        expl_outputs=model(**inputs['expl'])

        loss=self.alpha*pred_outputs.loss+(1-self.alpha)*expl_outputs.loss
        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        执行模型的预测步骤。

        参数：
        - model: nn.Module，Seq2Seq 模型。
        - inputs: Dict[str, Union[torch.Tensor, Any]]，包含 'pred' 和 'expl' 两个任务的输入数据。
        - prediction_loss_only: bool，是否仅计算预测损失。
        - ignore_keys: Optional[List[str]]，需要忽略的键列表。

        返回：
        - Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]，包括：
          1. 损失值（Optional[float]）。
          2. 预测 logits 列表，包括 'pred' 和 'expl' 两个任务的 logits。
          3. 真实标签列表，包括 'pred' 和 'expl' 两个任务的真实标签。
        """

        # 使用父类 Seq2SeqTrainer 的 prediction_step 方法对预测任务 'pred' 进行处理
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False,
                                               ignore_keys=ignore_keys)

        # 如果需要输出解释任务 'expl' 的结果
        if self.output_rationle:
            # 使用父类 Seq2SeqTrainer 的 prediction_step 方法对解释任务 'expl' 进行处理
            expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False,
                                                   ignore_keys=ignore_keys)
        else:
            # 否则，使用预测任务的输出作为占位符
            expl_outputs = pred_outputs  # 仅用作占位符

        # 计算加权损失，其中 alpha 为权重参数
        loss = self.alpha * pred_outputs[0] + (1 - self.alpha) * expl_outputs[0]

        # 返回损失值、预测 logits 列表和真实标签列表
        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )