from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
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
import os

############### 超参数设置 ###############
LR = 1e-5  # 降低学习率以提高稳定性
BATCH_SIZE = 8  # 保持小批次
GRAD_ACCUMULATION = 32  # 保持梯度累积
SEQ_LENGTH = 128  # 保持序列长度
TEMPERATURE = 1.0  # 降低温度系数，减少软标签的平滑程度
ALPHA = 0.5  # 降低蒸馏权重，增加原始任务权重
USE_8BIT = True
USE_4BIT = False
USE_GRADIENT_CHECKPOINTING = True

# 训练参数
NUM_EPOCHS = 100  # 保持轮数
SAVE_STEPS = 500  # 增加保存间隔
EVAL_STEPS = 500  # 增加评估间隔
WARMUP_STEPS = 1000  # 显著增加预热步数
MAX_GRAD_NORM = 0.3  # 进一步降低梯度裁剪阈值
#######################################

PATH = Path("./")
TEACHER_MODEL_NAME = "D:\ZLCODE\model\Llama-2-7b-chat-hf"
STUDENT_MODEL_NAME = "D:\ZLCODE\model\\"+"vicuna-7b-v1.5"
MODEL_OUTPUT = PATH / 'models/vicuna-7b-distilled'
TEACHER_LOGITS_PATH = PATH / 'teacher_logits.npy'

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

class CustomDataset(Dataset):
    def __init__(self, targets, goals, tokenizer, max_length, teacher_logits=None, generating_teacher_logits=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_logits = teacher_logits
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
            # 确保teacher_logits的形状正确
            if isinstance(self.teacher_logits, np.ndarray):
                item['teacher_logits'] = torch.tensor(self.teacher_logits[idx])
            else:
                item['teacher_logits'] = self.teacher_logits[idx]
        elif not self.generating_teacher_logits:  # 只在非生成阶段显示警告
            print(f"Warning: teacher_logits is None for index {idx}")
        
        return item

class SubsetDatasetWithLogits(Dataset):
    """保持teacher_logits的数据集子集"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]
        if 'teacher_logits' not in item:
            print(f"Warning: teacher_logits missing in SubsetDatasetWithLogits for idx {idx} (real_idx {real_idx})")
        return item

    def __len__(self):
        return len(self.indices)

def custom_collate_fn(batch):
    """自定义的数据整理函数，确保teacher_logits被正确处理"""
    # 初始化结果字典
    batch_dict = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'teacher_logits': []
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

class DistillationTrainer(Trainer):
    def __init__(self, *args, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        print(f"DistillationTrainer initialized with temperature={temperature}, alpha={alpha}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算蒸馏损失"""
        # 获取教师logits
        teacher_logits = inputs.pop("teacher_logits", None)
        if teacher_logits is None:
            print("Warning: teacher_logits is None in compute_loss")
            if return_outputs:
                return 0.0, None
            return 0.0
        
        # 获取学生模型输出
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # 计算原始任务的loss（语言模型loss）
        labels = inputs["labels"]
        lm_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100  # 忽略padding
        )
        
        # 计算蒸馏loss（KL散度）
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2)
        
        # 总loss为两部分的加权和
        loss = (1 - self.alpha) * lm_loss + self.alpha * distill_loss
        
        # 记录详细的loss信息
        if self.state.global_step % 10 == 0:  # 每10步记录一次详细信息
            self.log({
                "train/lm_loss": lm_loss.item(),
                "train/distill_loss": distill_loss.item(),
                "train/total_loss": loss.item(),
                "train/temperature": self.temperature,
                "train/alpha": self.alpha
            })
        
        if return_outputs:
            return loss, outputs
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """重写评估方法，添加蒸馏特定的指标"""
        # 运行原始的评估
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 计算额外的蒸馏指标
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        total_kl_div = 0
        total_js_div = 0
        total_top1_acc = 0
        total_top5_acc = 0
        n_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                teacher_logits = batch.pop("teacher_logits", None)
                if teacher_logits is None:
                    continue
                
                outputs = self.model(**batch)
                student_logits = outputs.logits
                
                # KL散度
                student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
                kl_div = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
                
                # JS散度
                student_probs_raw = F.softmax(student_logits, dim=-1)
                teacher_probs_raw = F.softmax(teacher_logits, dim=-1)
                m = 0.5 * (student_probs_raw + teacher_probs_raw)
                js_div = 0.5 * (
                    F.kl_div(torch.log(student_probs_raw), m, reduction='batchmean') +
                    F.kl_div(torch.log(teacher_probs_raw), m, reduction='batchmean')
                )
                
                # Top-1和Top-5准确率
                student_preds = student_logits.argmax(dim=-1)
                teacher_preds = teacher_logits.argmax(dim=-1)
                top1_acc = (student_preds == teacher_preds).float().mean()
                
                _, student_top5 = student_logits.topk(5, dim=-1)
                top5_acc = torch.any(student_top5 == teacher_preds.unsqueeze(-1), dim=-1).float().mean()
                
                batch_size = batch["input_ids"].size(0)
                total_kl_div += kl_div.item() * batch_size
                total_js_div += js_div.item() * batch_size
                total_top1_acc += top1_acc.item() * batch_size
                total_top5_acc += top5_acc.item() * batch_size
                n_samples += batch_size
        
        if n_samples > 0:
            metrics.update({
                f"{metric_key_prefix}/kl_divergence": total_kl_div / n_samples,
                f"{metric_key_prefix}/js_divergence": total_js_div / n_samples,
                f"{metric_key_prefix}/top1_accuracy": total_top1_acc / n_samples,
                f"{metric_key_prefix}/top5_accuracy": total_top5_acc / n_samples
            })
        
        self.log(metrics)
        return metrics

    def get_train_dataloader(self):
        """重写获取训练数据加载器的方法"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        # 使用自定义的collate_fn
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=custom_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """重写获取验证数据加载器的方法"""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=self._get_eval_sampler(eval_dataset),
            collate_fn=custom_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _prepare_inputs(self, data):
        """准备输入数据，确保teacher_logits被正确处理"""
        inputs = super()._prepare_inputs(data)
        if "teacher_logits" in data:
            inputs["teacher_logits"] = data["teacher_logits"].to(self.args.device)
        return inputs

def generate_teacher_logits(teacher_model):
    """生成并保存teacher模型的logits"""
    print("Generating teacher logits...")
    try:
        # 加载数据
        data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
        targets = data['target'].tolist()
        goals = data['goal'].tolist()
        
        temp_dataset = CustomDataset(targets, goals, tokenizer, SEQ_LENGTH, generating_teacher_logits=True)
        dataloader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(teacher_model.device)
                attention_mask = batch['attention_mask'].to(teacher_model.device)
                
                outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu().numpy()
                all_logits.append(logits)
        
        # 合并所有logits
        all_logits = np.concatenate(all_logits, axis=0)
        
        # 保存logits
        np.save(TEACHER_LOGITS_PATH, all_logits)
        print(f"Teacher logits saved to {TEACHER_LOGITS_PATH}")
        
    except Exception as e:
        print(f"生成teacher logits时发生错误: {str(e)}")
        raise

def prepare_datasets(teacher_logits):
    """准备训练和评估数据集"""
    # 加载数据
    data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    global targets, goals
    targets = data['target'].tolist()
    goals = data['goal'].tolist()
    
    # 准备数据集
    dataset = CustomDataset(targets, goals, tokenizer, SEQ_LENGTH, teacher_logits)
    print(f"总数据集大小: {len(dataset)}")
    
    # 验证第一个样本
    first_item = dataset[0]
    print("\n原始数据集第一个样本:")
    print("Keys:", first_item.keys())
    for k, v in first_item.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape:", v.shape)
    
    # 分割数据集
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
            # 直接使用数据集返回的编码后的数据
            inputs = {k: v.unsqueeze(0).to(student_model.device) for k, v in item.items() 
                     if k not in ['labels', 'teacher_logits']}
            student_outputs = student_model(**inputs)
            student_logits = student_outputs.logits
            
            # 获取教师模型的logits
            teacher_logits_batch = torch.tensor(teacher_logits[idx]).to(student_model.device)
            
            # 计算KL散度
            student_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
            teacher_probs = F.softmax(teacher_logits_batch / TEMPERATURE, dim=-1)
            kl_div = F.kl_div(
                student_probs,
                teacher_probs,
                reduction='batchmean',
            ) * (TEMPERATURE ** 2)
            
            # 计算交叉熵损失
            ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), 
                                    inputs["input_ids"].view(-1))
            
            # 计算准确率
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
    
    # 生成或加载teacher logits
    if not TEACHER_LOGITS_PATH.exists():
        print("Generating teacher logits...")
        # 临时加载教师模型生成logits
        teacher_model = LlamaForCausalLM.from_pretrained(
            TEACHER_MODEL_NAME,
            load_in_8bit=USE_8BIT,
            load_in_4bit=USE_4BIT,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        if USE_GRADIENT_CHECKPOINTING:
            teacher_model.gradient_checkpointing_enable()
            
        generate_teacher_logits(teacher_model)
        # 释放教师模型内存
        del teacher_model
        torch.cuda.empty_cache()
        print("Teacher logits generated and saved.")
    
    # 加载已保存的teacher logits
    teacher_logits = np.load(TEACHER_LOGITS_PATH)
    print("Teacher logits loaded from file.")
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_datasets(teacher_logits)
    
    # 现在只加载学生模型用于训练
    print("\n=== 模型加载信息 ===")
    print(f"模型输出目录: {MODEL_OUTPUT}")
    
    # 获取最新的检查点
    latest_checkpoint = get_latest_checkpoint()
    
    if latest_checkpoint is not None:
        model_path = os.path.join(latest_checkpoint, "complete_model")
        print(f"正在从检查点加载模型: {model_path}")
        student_model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=USE_8BIT,
            load_in_4bit=USE_4BIT,
            device_map="auto",
            torch_dtype=torch.float16,
            generation_config=generation_config
        )
    else:
        print(f"没有找到检查点，从原始模型加载: {STUDENT_MODEL_NAME}")
        student_model = LlamaForCausalLM.from_pretrained(
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
    run_name = f"distill-llama2-vicuna-t{TEMPERATURE}-a{ALPHA}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存最终模型
    print("\n=== 保存模型 ===")
    final_model_path = MODEL_OUTPUT / "final_model"
    print(f"基础保存路径: {final_model_path}")
    
    # 检查该目录是否已经存在，若存在则添加数字后缀
    i = 1
    while True:
        if i == 1:
            versioned_model_path = final_model_path
        else:
            versioned_model_path = final_model_path.parent / f"final_model-{i}"
        
        if not versioned_model_path.exists():
            break
        i += 1

    print(f"将使用的保存路径: {versioned_model_path}")
    versioned_model_path.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 保存完整模型（合并LoRA权重）
        print("1. 尝试保存完整模型（合并LoRA权重）...")
        
        try:
            # 首先尝试直接保存合并后的模型
            print("方法1: 直接保存合并后的模型...")
            merged_model = student_model.merge_and_unload()
            merged_model.save_pretrained(
                versioned_model_path / "complete_model",
                safe_serialization=True,
                max_shard_size="2GB"
            )
            print("- 完整模型已成功保存（方法1）")
            del merged_model
            torch.cuda.empty_cache()
            
        except Exception as e1:
            print(f"方法1失败: {str(e1)}")
            print("尝试方法2: 转换为float16后保存...")
            
            try:
                # 尝试将模型转换为float16后保存
                merged_model = student_model.merge_and_unload()
                merged_model = merged_model.half()  # 转换为float16
                merged_model.save_pretrained(
                    versioned_model_path / "complete_model",
                    safe_serialization=True,
                    max_shard_size="2GB"
                )
                print("- 完整模型已成功保存（方法2）")
                del merged_model
                torch.cuda.empty_cache()
                
            except Exception as e2:
                print(f"方法2失败: {str(e2)}")
                print("尝试方法3: 使用state_dict保存...")
                
                try:
                    # 尝试只保存状态字典
                    merged_model = student_model.merge_and_unload()
                    torch.save(
                        merged_model.state_dict(),
                        versioned_model_path / "complete_model" / "pytorch_model.bin"
                    )
                    print("- 模型状态字典已成功保存（方法3）")
                    del merged_model
                    torch.cuda.empty_cache()
                    
                except Exception as e3:
                    print(f"所有保存方法都失败。最后一个错误: {str(e3)}")
                    print("将只保存LoRA权重和配置文件")
    
    except Exception as e:
        print(f"合并模型时出错: {str(e)}")
        print("继续保存其他文件...")
    
    # 2. 单独保存 LoRA 适配器
    print("2. 保存 LoRA 适配器...")
    try:
        student_model.save_pretrained(versioned_model_path / "lora_adapter")
        print("- LoRA适配器保存成功")
    except Exception as e:
        print(f"保存LoRA适配器时出错: {str(e)}")
    
    # 3. 保存分词器
    print("3. 保存分词器...")
    try:
        tokenizer.save_pretrained(versioned_model_path)
        print("- 分词器保存成功")
    except Exception as e:
        print(f"保存分词器时出错: {str(e)}")
    
    # 4. 保存生成配置
    print("4. 保存生成配置...")
    try:
        generation_config = get_generation_config()
        generation_config.save_pretrained(versioned_model_path)
        print("- 生成配置保存成功")
    except Exception as e:
        print(f"保存生成配置时出错: {str(e)}")
    
    # 5. 保存训练配置和参数
    print("5. 保存训练配置...")
    try:
        training_config = {
            "teacher_model": TEACHER_MODEL_NAME,
            "student_model": STUDENT_MODEL_NAME,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQ_LENGTH,
            "num_epochs": NUM_EPOCHS,
            "gradient_accumulation": GRAD_ACCUMULATION,
            "training_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(versioned_model_path / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=4, ensure_ascii=False)
        print("- 训练配置保存成功")
    except Exception as e:
        print(f"保存训练配置时出错: {str(e)}")
    
    print("\n模型保存完成！")
    print(f"保存位置: {versioned_model_path}")
    print("保存内容:")
    print("- 完整模型（合并LoRA权重）: /complete_model")
    print("- LoRA适配器: /lora_adapter")
    print("- 分词器")
    print("- 生成配置")
    print("- 训练配置")
    print("================\n")
    
    # 加载原始模型进行比较（不使用量化）
    print("Loading original model for comparison...")
    original_model = LlamaForCausalLM.from_pretrained(
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
