import json
import numpy as np
import glob
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# 定义类别
labels = ["Real", "Fake"]
label_mapping = {label: i for i, label in enumerate(labels)}

def calculate_model_performance(data, output_key):
    """
    计算模型跑分和混淆矩阵
    
    参数:
    data: list of dict, 每个字典包含 'info_type'（真实标签）和 'output_key'（预测标签）
    
    返回:
    包含各类指标和混淆矩阵的字典
    """
    
    # 初始化列表存储真实标签和预测标签
    y_true = []
    y_pred = []
    
    # 处理数据
    for item in data:
        true_label = item['info_type']
        if output_key in item:
            pred_label = item[output_key].strip().split(' ')[0]
        else:
            continue
        
        # 确保真实标签在有效类别中
        if true_label not in labels:
            continue
            
        # 转换真实标签为索引
        y_true.append(label_mapping[true_label])
        
        # 处理预测标签：如果在有效类别中则使用，否则标记为"未知"
        if pred_label in labels:
            y_pred.append(label_mapping[pred_label])
        else:
            # 对于不在标签中的预测，我们可以将其视为错误分类
            # 这里我们将其映射到一个特殊值-1，后续在混淆矩阵中处理
            y_pred.append(-1)
    
    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 统计预测分布
    pred_distribution = {}
    for i, label in enumerate(labels):
        pred_distribution[label] = np.sum(y_pred == i)
    pred_distribution["Unknown"] = np.sum(y_pred == -1)
    
    # 计算基础指标
    results = {
        'total_samples': len(y_true),
        'prediction_distribution': pred_distribution
    }
    
    # 计算准确率（只考虑预测在有效标签中的样本）
    valid_mask = y_pred != -1
    if np.sum(valid_mask) > 0:
        valid_accuracy = accuracy_score(y_true[valid_mask], y_pred[valid_mask])
        results['accuracy_valid_only'] = valid_accuracy
    
    # 计算包含未知预测的准确率（将未知视为错误预测）
    correct_predictions = np.sum((y_pred != -1) & (y_pred == y_true))
    accuracy_with_unknown = correct_predictions / len(y_true)
    results['accuracy_with_unknown'] = accuracy_with_unknown
    
    # 计算混淆矩阵（只考虑有效预测）
    if np.sum(valid_mask) > 0:
        cm = confusion_matrix(y_true[valid_mask], y_pred[valid_mask], labels=range(len(labels)))
        
        # 计算每个类别的指标
        precision = precision_score(y_true[valid_mask], y_pred[valid_mask], average=None, zero_division=0)
        recall = recall_score(y_true[valid_mask], y_pred[valid_mask], average=None, zero_division=0)
        f1 = f1_score(y_true[valid_mask], y_pred[valid_mask], average=None, zero_division=0)
        
        # 计算宏平均
        macro_precision = precision_score(y_true[valid_mask], y_pred[valid_mask], average='macro', zero_division=0)
        macro_recall = recall_score(y_true[valid_mask], y_pred[valid_mask], average='macro', zero_division=0)
        macro_f1 = f1_score(y_true[valid_mask], y_pred[valid_mask], average='macro', zero_division=0)
        
        # 计算微平均
        micro_precision = precision_score(y_true[valid_mask], y_pred[valid_mask], average='micro', zero_division=0)
        micro_recall = recall_score(y_true[valid_mask], y_pred[valid_mask], average='micro', zero_division=0)
        micro_f1 = f1_score(y_true[valid_mask], y_pred[valid_mask], average='micro', zero_division=0)
        
        # 计算加权平均
        weighted_precision = precision_score(y_true[valid_mask], y_pred[valid_mask], average='weighted', zero_division=0)
        weighted_recall = recall_score(y_true[valid_mask], y_pred[valid_mask], average='weighted', zero_division=0)
        weighted_f1 = f1_score(y_true[valid_mask], y_pred[valid_mask], average='weighted', zero_division=0)
        
        # 整理结果
        results.update({
            'confusion_matrix': cm.tolist(),
            'class_metrics': {
                label: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i])
                }
                for i, label in enumerate(labels)
            },
            'macro_avg': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1_score': float(macro_f1)
            },
            'micro_avg': {
                'precision': float(micro_precision),
                'recall': float(micro_recall),
                'f1_score': float(micro_f1)
            },
            'weighted_avg': {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1_score': float(weighted_f1)
            }
        })
    else:
        results['confusion_matrix'] = None
        results['class_metrics'] = None
        results['macro_avg'] = None
        results['micro_avg'] = None
        results['weighted_avg'] = None
    
    return results


def print_confusion_matrix(cm, labels, output_path):
    """美化打印混淆矩阵"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*10)
        f.write("混淆矩阵 (Confusion Matrix):")
        f.write("="*10)
        
        # 确定列宽
        col_width = max(len(label) for label in labels) + 2
        
        # 打印表头
        f.write(f"{'真实/预测':<{col_width}}", end="")
        for label in labels:
            f.write(f"{label:<{col_width}}", end="")
        f.write("总计")
        f.write("-" * (col_width * (len(labels) + 2)))
        
        # 打印每一行
        for i, true_label in enumerate(labels):
            f.write(f"{true_label:<{col_width}}", end="")
            row_sum = 0
            for j in range(len(labels)):
                f.write(f"{cm[i][j]:<{col_width}}", end="")
                row_sum += cm[i][j]
            f.write(f"{row_sum}")
        
        # 打印列总计
        f.write("-" * (col_width * (len(labels) + 2)))
        f.write(f"{'总计':<{col_width}}", end="")
        col_sums = []
        for j in range(len(labels)):
            col_sum = sum(cm[i][j] for i in range(len(labels)))
            col_sums.append(col_sum)
            f.write(f"{col_sum:<{col_width}}", end="")
        f.write(f"{sum(col_sums)}")
        
        # 计算并打印准确率（对角线和除以总数）
        diagonal_sum = sum(cm[i][i] for i in range(len(labels)))
        total_sum = sum(sum(row) for row in cm)
        if total_sum > 0:
            accuracy = diagonal_sum / total_sum
            f.write(f"\n整体准确率: {accuracy:.4f} ({diagonal_sum}/{total_sum})")


def print_detailed_metrics(results, output_path, mode):
    """打印详细指标"""
    with open(output_path, mode, encoding='utf-8') as f:
        f.write("\n" + "="*10)
        f.write("模型性能指标:")
        f.write("="*10)
        
        f.write(f"\n样本统计:")
        f.write(f"  总样本数: {results['total_samples']}")
        f.write(f"  有效预测样本数: {results['total_samples'] - results['prediction_distribution']['Unknown']}")
        f.write(f"  未知预测数: {results['prediction_distribution']['Unknown']}")
        
        f.write(f"\n预测分布:")
        for label, count in results['prediction_distribution'].items():
            f.write(f"  {label}: {count} ({count/results['total_samples']*100:.1f}%)")
        
        if 'accuracy_with_unknown' in results:
            f.write(f"\n准确率指标:")
            f.write(f"  包含未知预测的准确率: {results['accuracy_with_unknown']:.4f}")
            if 'accuracy_valid_only' in results:
                f.write(f"  仅有效预测的准确率: {results['accuracy_valid_only']:.4f}")
        
        if results['class_metrics']:
            f.write(f"\n各类别详细指标:")
            f.write("-"*10)
            for label, metrics in results['class_metrics'].items():
                f.write(f"\n{label}:")
                f.write(f"  精确率 (Precision): {metrics['precision']:.4f}")
                f.write(f"  召回率 (Recall): {metrics['recall']:.4f}")
                f.write(f"  F1分数: {metrics['f1_score']:.4f}")
            
            f.write(f"\n{'='*10}")
            f.write("综合指标:")
            f.write(f"  宏平均精确率: {results['macro_avg']['precision']:.4f}")
            f.write(f"  宏平均召回率: {results['macro_avg']['recall']:.4f}")
            f.write(f"  宏平均F1分数: {results['macro_avg']['f1_score']:.4f}")
            f.write(f"\n  微平均精确率: {results['micro_avg']['precision']:.4f}")
            f.write(f"  微平均召回率: {results['micro_avg']['recall']:.4f}")
            f.write(f"  微平均F1分数: {results['micro_avg']['f1_score']:.4f}")
            f.write(f"\n  加权平均精确率: {results['weighted_avg']['precision']:.4f}")
            f.write(f"  加权平均召回率: {results['weighted_avg']['recall']:.4f}")
            f.write(f"  加权平均F1分数: {results['weighted_avg']['f1_score']:.4f}")
            
            f.write('\n\n')


# 示例使用
if __name__ == "__main__":
    files = glob.glob('chat/data/emineyetm_fake-news-detection-datasets/FakeNews_*.json')
    for file_path in files:
        model_name = Path(file_path).stem.replace('FakeNews_', '')
        with open(file_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
    
        # 计算性能指标
        results1 = calculate_model_performance(sample_data, f'answer_base_{model_name}')
        results2 = calculate_model_performance(sample_data, f'answer_disorder_{model_name}')
        results3 = calculate_model_performance(sample_data, f'answer_disorder_intent_{model_name}')
        
        # 打印结果
        output_prefix = file_path[:-5]
        print(results1)
        print_detailed_metrics(results1, output_prefix + '_result.txt', 'w')
        # print_confusion_matrix(results1['confusion_matrix'], labels, output_prefix + '_base_cm.txt')  
        print(results2)          
        print_detailed_metrics(results2, output_prefix + '_result.txt', 'a')
        # print_confusion_matrix(results2['confusion_matrix'], labels, output_prefix + '_disorder_cm.txt')
        print(results3)
        print_detailed_metrics(results3, output_prefix + '_result.txt', 'a')
        # print_confusion_matrix(results3['confusion_matrix'], labels, output_prefix + '_disorder_intent_cm.txt')
        
        