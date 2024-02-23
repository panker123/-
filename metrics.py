import numpy as np

def compute_text_acc(preds,labels):
    return np.mean(np.array(preds)==np.array(labels))


def compute_equation_acc(preds,labels):
    preds=[eval_equation(pred)for pred in preds]
    labels=[eval_equation(label)for label in labels]

    return np.mean(np.array(preds)==np.array(labels))

def eval_equation(equation):
    try:
        answer=eval(equation)
    except:
        answer=np.nan

    return answer


def compute_metrics_text(tokenizer):
    # 定义计算模型指标的方法
    def compute_metrics(eval_pred):
        # 从评估预测中获取模型的预测和真实标签
        predictions, labels = eval_pred
        # 使用 tokenizer 将模型的预测解码成文本
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        # 将真实标签中为 -100 的部分替换成 tokenizer 的 pad_token_id
        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        # 使用 tokenizer 将真实标签解码成文本
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 计算准确率
        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        # 返回计算得到的指标（这里仅包括准确率）
        return {'accuracy': acc}

    # 返回计算指标的方法
    return compute_metrics


def compute_metrics_text_aux(tokenizer):
    # 定义计算模型指标的方法
    def compute_metrics(eval_pred):
        # 从评估预测中获取模型的预测和真实标签
        predictions, labels = eval_pred
        # 使用 tokenizer 将模型的预测解码成文本
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # 将真实标签中为 -100 的部分替换成 tokenizer 的 pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # 使用 tokenizer 将真实标签解码成文本
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 计算准确率
        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        # 返回计算得到的指标（这里仅包括准确率）
        return {'accuracy': acc}

    # 返回计算指标的方法
    return compute_metrics


def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}

    return compute_metrics


def compute_metrics_equation_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}

    return compute_metrics