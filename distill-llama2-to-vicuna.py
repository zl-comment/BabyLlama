from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import wandb
import json
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datetime

############### 超参数设置 ###############
LR = 1e-5  # 降低学习率
BATCH_SIZE = 18  # 根据显存调整
GRAD_ACCUMULATION = 30  # 增加梯度累积步数
SEQ_LENGTH = 128
TEMPERATURE = 2.0
ALPHA = 0.5  # 蒸馏权重
ATT_ALPHA = 0.1  # 注意力对齐损失的权重
USE_8BIT = True
USE_4BIT = False
USE_GRADIENT_CHECKPOINTING = True

# 训练参数
NUM_EPOCHS = 50  # 增加训练轮数
SAVE_STEPS = 100  # 增加保存间隔
EVAL_STEPS = 100  # 增加评估间隔
WARMUP_STEPS = 100  # 增加预热步数
MAX_GRAD_NORM = 1.0  # 降低梯度裁剪阈值
#######################################

PATH = Path("./")
TEACHER_MODEL_NAME = "D:/ZLCODE/model/Llama-2-7b-chat-hf"
STUDENT_MODEL_NAME = "D:/ZLCODE/model/vicuna-7b-v1.5"
MODEL_OUTPUT = PATH / 'models/vicuna-7b-distilled'
TEACHER_LOGITS_PATH = PATH / 'teacher_logits.npy'
TEACHER_ATT_PATH = PATH / 'teacher_attentions.npy'

# Wandb配置
wandb_log = True  # 是否使用wandb记录训练日志


def get_generation_config():
    """获取生成配置"""
    return GenerationConfig(
        do_sample=True,  # 启用采样
        temperature=0.9,
        top_p=0.6,
        max_length=SEQ_LENGTH,
        pad_token_id=0
    )


def get_latest_checkpoint():
    """获取最新的检查点目录"""
    if not MODEL_OUTPUT.exists():
        print(f"模型输出目录不存在: {MODEL_OUTPUT}")
        return None

    checkpoints = [d for d in MODEL_OUTPUT.iterdir() if d.is_dir() and d.name.startswith('final_model-')]
    if not checkpoints:
        print(f"在 {MODEL_OUTPUT} 中没有找到检查点")
        return None

    # 从检查点名称中提取步数，并找到最大的
    checkpoint_steps = [int(cp.name.split('-')[-1]) for cp in checkpoints]
    latest_step = max(checkpoint_steps)
    latest_checkpoint = MODEL_OUTPUT / f'final_model-{latest_step}'

    print("\n=== 检查点信息 ===")
    print(f"检查点目录: {MODEL_OUTPUT}")
    print("所有检查点:")
    for step in sorted(checkpoint_steps):
        checkpoint = MODEL_OUTPUT / f'final_model-{step}'
        if checkpoint == latest_checkpoint:
            print(f" - {checkpoint} (最新)")
        else:
            print(f" - {checkpoint}")
    print("================\n")

    return latest_checkpoint

class SubsetDatasetWithLogits(Dataset):
    """保持teacher_logits和teacher_attentions的数据集子集"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]
        if 'teacher_logits' not in item:
            print(f"Warning: teacher_logits missing in SubsetDatasetWithLogits for idx {idx} (real_idx {real_idx})")
        if 'teacher_attentions' not in item:
            print(f"Warning: teacher_attentions missing in SubsetDatasetWithLogits for idx {idx} (real_idx {real_idx})")
        return item

    def __len__(self):
        return len(self.indices)


def custom_collate_fn(batch):
    """自定义的数据整理函数，确保teacher_logits和teacher_attentions被正确处理"""
    # 初始化结果字典
    batch_dict = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'teacher_logits': [],
        'teacher_attentions': []
    }

    # 收集每个样本的数据
    for item in batch:
        for key in batch_dict:
            if key in item:
                batch_dict[key].append(item[key])

    # 将列表转换为张量
    for key in batch_dict:
        if batch_dict[key]:
            batch_dict[key] = torch.stack(batch_dict[key])

    return batch_dict


class CustomDataset(Dataset):
    def __init__(self, targets, goals, tokenizer, max_length, teacher_logits=None, teacher_attentions=None,
                 generating_teacher_logits=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_logits = teacher_logits
        self.teacher_attentions = teacher_attentions
        self.generating_teacher_logits = generating_teacher_logits

        # 组合target和goal成对话格式
        self.conversations = []
        for target, goal in zip(targets, goals):
            prompt = f"Human: Given the goal: {goal}\nAssistant: {target}"
            self.conversations.append(prompt)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.conversations[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = item['input_ids'].clone()

        if self.teacher_logits is not None:
            # 确保teacher_logits的形状和类型正确
            if isinstance(self.teacher_logits, np.ndarray):
                item['teacher_logits'] = torch.tensor(self.teacher_logits[idx], dtype=torch.float32)
            else:
                item['teacher_logits'] = self.teacher_logits[idx].float()

        if self.teacher_attentions is not None:
            # 确保teacher_attentions的形状和类型正确
            if isinstance(self.teacher_attentions, np.ndarray):
                item['teacher_attentions'] = torch.tensor(self.teacher_attentions[idx], dtype=torch.float32)
            else:
                item['teacher_attentions'] = self.teacher_attentions[idx].float()
        elif not self.generating_teacher_logits:  # 只在非生成阶段显示警告
            print(f"Warning: teacher_attentions is None for index {idx}")

        return item

class DistillationTrainer(Trainer):
    def __init__(self, *args, temperature=2.0, alpha=0.5, att_alpha=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.att_alpha = att_alpha  # 注意力对齐损失的权重
        print(f"DistillationTrainer initialized with temperature={temperature}, alpha={alpha}, att_alpha={att_alpha}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算蒸馏损失，包括语言模型损失、蒸馏损失和注意力对齐损失
        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
            kwargs: 其他参数
        """
        # 获取教师logits和attentions
        teacher_logits = inputs.pop("teacher_logits", None)
        teacher_attentions = inputs.pop("teacher_attentions", None)

        # 获取学生模型输出，包括注意力权重
        outputs = model(**inputs, output_attentions=True)
        student_logits = outputs.logits
        student_attentions = outputs.attentions  # 列表，每个元素对应一个层的注意力权重

        # 计算语言模型的交叉熵损失
        labels = inputs["labels"]
        lm_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        # 计算蒸馏损失（KL散度）
        if teacher_logits is not None:
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            distill_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
        else:
            distill_loss = 0.0

        # 计算注意力对齐损失
        if teacher_attentions is not None:
            # 这里假设教师模型和学生模型有相同数量的层和注意力头
            att_loss = 0.0
            for t_layer, s_layer in zip(teacher_attentions, student_attentions):
                # 不对教师注意力权重进行平均，保持形状一致
                teacher_att = t_layer.float()  # [batch_size, seq_length, seq_length]
                student_att = s_layer.mean(dim=1).float()  # [batch_size, seq_length, seq_length]

                # 打印形状和数据类型以调试
                print(f"teacher_att shape: {teacher_att.shape}, dtype: {teacher_att.dtype}")
                print(f"student_att shape: {student_att.shape}, dtype: {student_att.dtype}")

                # 计算MSE损失
                att_loss += F.mse_loss(student_att, teacher_att)

            # 平均所有层的注意力损失
            att_loss /= len(student_attentions)
        else:
            att_loss = 0.0

        # 综合损失
        loss = (1 - self.alpha) * lm_loss + self.alpha * distill_loss + self.att_alpha * att_loss

        if return_outputs:
            return loss, outputs
        return loss

    def _prepare_inputs(self, data):
        """准备输入数据，确保teacher_logits和teacher_attentions被正确处理"""
        inputs = super()._prepare_inputs(data)
        if "teacher_logits" in data:
            inputs["teacher_logits"] = data["teacher_logits"].to(self.args.device)
        if "teacher_attentions" in data:
            inputs["teacher_attentions"] = data["teacher_attentions"].to(self.args.device)
        return inputs

def generate_teacher_logits_and_attentions(teacher_model):
    """生成并保存教师模型的 logits 和注意力权重"""
    print("Generating teacher logits and attentions...")
    try:
        # 加载数据
        data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
        targets = data['target'].tolist()
        goals = data['goal'].tolist()

        # 创建临时数据集，不包含 teacher_logits 和 teacher_attentions
        temp_dataset = CustomDataset(targets, goals, tokenizer, SEQ_LENGTH, generating_teacher_logits=True)
        dataloader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_logits = []
        all_attentions = []

        # 确保模型处于评估模式
        teacher_model.eval()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(teacher_model.device)
                attention_mask = batch['attention_mask'].to(teacher_model.device)

                # 获取模型输出，包括注意力权重
                outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                logits = outputs.logits.cpu().numpy()  # [batch_size, seq_length, vocab_size]
                attentions = outputs.attentions  # List of tensors: one per layer

                # 选择所有层的注意力权重，并平均所有头
                # 形状：[batch_size, num_heads, seq_length, seq_length] -> [batch_size, seq_length, seq_length]
                avg_attentions = [att.mean(dim=1).cpu().numpy() for att in attentions]

                # 将所有层的平均注意力权重进行堆叠，形状：[num_layers, batch_size, seq_length, seq_length]
                # 然后再合并所有批次，最终形状：[total_samples, num_layers, seq_length, seq_length]
                avg_attentions = np.stack(avg_attentions, axis=0)  # [num_layers, batch_size, seq_length, seq_length]
                avg_attentions = np.transpose(avg_attentions, (1, 0, 2, 3))  # [batch_size, num_layers, seq_length, seq_length]

                all_logits.append(logits)
                all_attentions.append(avg_attentions)

        # 合并所有 logits 和 attentions
        all_logits = np.concatenate(all_logits, axis=0)  # [total_samples, seq_length, vocab_size]
        all_attentions = np.concatenate(all_attentions, axis=0)  # [total_samples, num_layers, seq_length, seq_length]

        # 保存 logits 和 attentions
        np.save(TEACHER_LOGITS_PATH, all_logits)
        np.save(TEACHER_ATT_PATH, all_attentions)
        print(f"Teacher logits saved to {TEACHER_LOGITS_PATH}")
        print(f"Teacher attentions saved to {TEACHER_ATT_PATH}")

    except Exception as e:
        print(f"生成teacher logits和attentions时发生错误: {str(e)}")
        raise

def prepare_datasets(teacher_logits, teacher_attentions):
    """准备训练和评估数据集"""
    # 加载数据
    data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    targets = data['target'].tolist()
    goals = data['goal'].tolist()

    # 创建完整的数据集，包括 logits 和 attentions
    dataset = CustomDataset(
        targets=targets,
        goals=goals,
        tokenizer=tokenizer,
        max_length=SEQ_LENGTH,
        teacher_logits=teacher_logits,
        teacher_attentions=teacher_attentions
    )
    print(f"总数据集大小: {len(dataset)}")

    # 验证第一个样本
    first_item = dataset[0]
    print("\n原始数据集第一个样本:")
    print("Keys:", first_item.keys())
    for k, v in first_item.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape:", v.shape)

    # 分割数据集为训练集和验证集
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    train_dataset = SubsetDatasetWithLogits(dataset, train_indices)
    eval_dataset = SubsetDatasetWithLogits(dataset, eval_indices)

    print(f"\n训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")

    # 验证训练集第一个样本
    first_train_item = train_dataset[0]
    print("\n训练集第一个样本:")
    print("Keys:", first_train_item.keys())
    for k, v in first_train_item.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape:", v.shape)

    return train_dataset, eval_dataset

def compute_distillation_metrics(student_model, teacher_logits, eval_dataset, tokenizer):
    """计算蒸馏相关的评估指标"""
    student_model.eval()
    total_kl_div = 0
    total_ce_loss = 0  # 交叉熵损失
    total_accuracy = 0  # 添加准确率指标
    n_samples = 0

    with torch.no_grad():
        for idx, item in enumerate(eval_dataset):
            # 准备输入数据
            inputs = {
                "input_ids": item["input_ids"].unsqueeze(0).to(student_model.device),
                "attention_mask": item["attention_mask"].unsqueeze(0).to(student_model.device)
            }
            labels = item["labels"].unsqueeze(0).to(student_model.device)

            # 获取学生模型输出
            outputs = student_model(**inputs, output_attentions=True)
            student_logits = outputs.logits  # [1, seq_length, vocab_size]
            student_attentions = outputs.attentions  # List of tensors: one per layer

            # 获取教师模型的 logits
            teacher_logits_batch = torch.tensor(teacher_logits[idx]).unsqueeze(0).to(student_model.device)  # [1, seq_length, vocab_size]

            # 计算 KL 散度
            teacher_probs = F.softmax(teacher_logits_batch / TEMPERATURE, dim=-1)
            student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
            kl_div = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (TEMPERATURE ** 2)

            # 计算交叉熵损失
            ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

            # 计算准确率（与教师模型预测一致）
            student_preds = student_logits.argmax(dim=-1)
            teacher_preds = teacher_logits_batch.argmax(dim=-1)
            accuracy = (student_preds == teacher_preds).float().mean()

            # 更新统计
            batch_size = inputs["input_ids"].size(0)
            total_kl_div += kl_div.item() * batch_size
            total_ce_loss += ce_loss.item() * batch_size
            total_accuracy += accuracy.item() * batch_size
            n_samples += batch_size

    metrics = {
        "kl_divergence": total_kl_div / n_samples,
        "cross_entropy_loss": total_ce_loss / n_samples,
        "prediction_accuracy": total_accuracy / n_samples,
    }

    return metrics


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # 获取生成配置
    generation_config = get_generation_config()

    # 生成或加载teacher logits和attentions
    if not TEACHER_LOGITS_PATH.exists() or not Path(TEACHER_ATT_PATH).exists():
        print("Generating teacher logits and attentions...")
        # 临时加载教师模型生成logits和attentions
        teacher_model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_NAME,
            load_in_8bit=USE_8BIT,
            load_in_4bit=USE_4BIT,
            device_map="auto",
            torch_dtype=torch.float16
        )

        if USE_GRADIENT_CHECKPOINTING:
            teacher_model.gradient_checkpointing_enable()

        generate_teacher_logits_and_attentions(teacher_model)
        # 释放教师模型内存
        del teacher_model
        torch.cuda.empty_cache()
        print("Teacher logits and attentions generated and saved.")

    # 加载已保存的teacher logits和attentions
    teacher_logits = np.load(TEACHER_LOGITS_PATH)
    teacher_attentions = np.load(TEACHER_ATT_PATH)
    print("Teacher logits and attentions loaded from file.")

    # 准备数据集
    train_dataset, eval_dataset = prepare_datasets(teacher_logits, teacher_attentions)

    # 现在只加载学生模型用于训练
    print("\n=== 模型加载信息 ===")
    print(f"模型输出目录: {MODEL_OUTPUT}")

    # 获取最新的检查点
    latest_checkpoint = get_latest_checkpoint()

    if latest_checkpoint is not None:
        print(f"正在从检查点加载模型: {latest_checkpoint}")
        student_model = AutoModelForCausalLM.from_pretrained(
            latest_checkpoint,
            load_in_8bit=USE_8BIT,
            load_in_4bit=USE_4BIT,
            device_map="auto",
            torch_dtype=torch.float16,
            generation_config=generation_config
        )
    else:
        print(f"没有找到检查点，从原始模型加载: {STUDENT_MODEL_NAME}")
        student_model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_NAME,
            load_in_8bit=USE_8BIT,
            load_in_4bit=USE_4BIT,
            device_map="auto",
            torch_dtype=torch.float16,
            generation_config=generation_config
        )
    print("==================\n")

    # 准备模型进行kbit训练
    student_model = prepare_model_for_kbit_training(student_model)

    if USE_GRADIENT_CHECKPOINTING:
        student_model.gradient_checkpointing_enable()

    # 配置LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # 将模型转换为PEFT模型
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # 配置训练参数
    output_dir = MODEL_OUTPUT
    run_name = f"distill-llama2-vicuna-t{TEMPERATURE}-a{ALPHA}-att{ATT_ALPHA}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT,
        run_name=run_name,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,  # 只保留最新的3个检查点
        load_best_model_at_end=True,
        report_to="wandb" if wandb_log else None,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        optim="adamw_torch",
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        resume_from_checkpoint=latest_checkpoint,  # 使用找到的最新检查点
    )

    # 初始化wandb
    if wandb_log:
        wandb.init(
            project="llama2-vicuna-distillation",
            name=run_name,
            config={
                "teacher_model": TEACHER_MODEL_NAME,
                "student_model": STUDENT_MODEL_NAME,
                "temperature": TEMPERATURE,
                "alpha": ALPHA,
                "att_alpha": ATT_ALPHA,
                "learning_rate": LR,
                "batch_size": BATCH_SIZE,
                "sequence_length": SEQ_LENGTH,
                "train_size": len(train_dataset),
                "eval_size": len(eval_dataset),
            }
        )

    # 初始化训练器
    trainer = DistillationTrainer(
        student_model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        temperature=TEMPERATURE,
        alpha=ALPHA,
        att_alpha=ATT_ALPHA,
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 保存最终模型
    print("\n=== 保存模型 ===")
    final_model_path = MODEL_OUTPUT / "final_model"
    print(f"保存路径: {final_model_path}")

    # 检查该目录是否已经存在，若存在则添加数字后缀
    i = 1
    new_model_path = final_model_path
    while new_model_path.exists():
        final_model_path = new_model_path.parent / f"final_model-{i}"
        i += 1
        print(final_model_path)

    # 保存 PEFT/LoRA 适配器
    print("1. 保存 LoRA 适配器...")
    student_model.save_pretrained(final_model_path)
    # 保存分词器
    print("2. 保存分词器...")
    tokenizer.save_pretrained(final_model_path)

    # 保存生成配置
    print("3. 保存生成配置...")
    generation_config = get_generation_config()
    generation_config.save_pretrained(final_model_path)

    print("模型保存完成！")
    print("================\n")

    # 加载原始模型进行比较（不使用量化）
    print("Loading original model for comparison...")
    original_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 评估并比较模型
    print("Evaluating models...")
    eval_results = compute_distillation_metrics(original_model, teacher_logits, eval_dataset, tokenizer)

    # 打印评估结果
    print("\nEvaluation Results:")
    print("Original Model:")
    print(f"KL Divergence: {eval_results['kl_divergence']:.4f}")
    print(f"Cross Entropy Loss: {eval_results['cross_entropy_loss']:.4f}")
    print(f"Prediction Accuracy: {eval_results['prediction_accuracy']:.4f}")

    # 保存评估结果
    with open(MODEL_OUTPUT / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    print("Training completed!")

if __name__ == "__main__":
    main()
