from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
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
from tqdm import tqdm

############### 超参数设置 ###############
LR = 2e-5  # 稍微提高学习率
BATCH_SIZE = 2  # 增加batch size
GRAD_ACCUMULATION = 2  # 减少梯度累积步数
SEQ_LENGTH = 512
TEMPERATURE = 2.0
ALPHA = 0.5
#######################################

PATH = Path("./")
TEACHER_MODEL_NAME = "D:\ZLCODE\model\Llama-2-7b-chat-hf"
STUDENT_MODEL_NAME = "D:\ZLCODE\model\\"+"vicuna-7b-v1.5"
MODEL_OUTPUT = PATH / 'models/vicuna-7b-distilled'
TEACHER_LOGITS_PATH = PATH / 'teacher_logits.npy'

wandb_log = True

class CustomDataset(Dataset):
    def __init__(self, targets, goals, tokenizer, max_length, teacher_logits=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_logits = teacher_logits
        
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
        else:
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
        self.progress_bar = None
        self.temperature = temperature
        self.alpha = alpha
        print(f"DistillationTrainer initialized with temperature={temperature}, alpha={alpha}")

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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=len(self.train_dataset), desc="Training progress")
        self.progress_bar.update(len(inputs['input_ids']))
        
        # 检查是否有teacher_logits
        if 'teacher_logits' not in inputs:
            print("\n缺少teacher_logits!")
            print("当前inputs的keys:", inputs.keys())
            print("当前batch的形状:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {v.shape}")
            raise ValueError("teacher_logits not found in inputs")
            
        # 移除teacher_logits从inputs
        teacher_logits = inputs.pop('teacher_logits')
        
        # 计算学生模型输出
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        
        # 计算蒸馏损失
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
            )
            * (self.temperature ** 2)
        )
        
        loss = self.alpha * student_loss + (1.0 - self.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

    def _prepare_inputs(self, data):
        """准备输入数据，确保teacher_logits被正确处理"""
        prepared = super()._prepare_inputs(data)
        if 'teacher_logits' in data:
            prepared['teacher_logits'] = data['teacher_logits'].to(self.args.device)
        return prepared

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            if self.progress_bar is not None:
                self.progress_bar.close()

def generate_teacher_logits():
    print("Generating teacher logits...")
    
    try:
        # 加载教师模型
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        teacher.eval()
        
        # 创建没有teacher_logits的数据集
        temp_dataset = CustomDataset(targets, goals, tokenizer, SEQ_LENGTH)
        dataloader = DataLoader(temp_dataset, batch_size=BATCH_SIZE)
        
        all_logits = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Generating teacher logits", total=len(dataloader))
            for batch in progress_bar:
                try:
                    batch = {k: v.to(teacher.device) for k, v in batch.items()}
                    outputs = teacher(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    logits = outputs.logits.cpu().numpy()
                    all_logits.extend(logits)
                    progress_bar.update(1)
                    # 主动清理内存
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("警告:GPU内存不足,跳过当前批次")
                        torch.cuda.empty_cache()
                        continue
                    raise e
        
        # 释放教师模型内存
        del teacher
        torch.cuda.empty_cache()
        
        # 保存logits
        np.save(TEACHER_LOGITS_PATH, np.array(all_logits))
        print("Teacher logits saved to", TEACHER_LOGITS_PATH)
        return np.array(all_logits)
    except Exception as e:
        print(f"生成teacher logits时发生错误: {str(e)}")
        raise

def main():
    # 加载数据
    data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    global targets, goals
    targets = data['target'].tolist()
    goals = data['goal'].tolist()
    
    # 加载tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 生成或加载teacher logits
    if not TEACHER_LOGITS_PATH.exists():
        teacher_logits = generate_teacher_logits()
    else:
        print("Loading existing teacher logits...")
        teacher_logits = np.load(TEACHER_LOGITS_PATH)
        print(f"Loaded teacher_logits shape: {teacher_logits.shape}")
    
    # 创建数据集
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
    
    # 初始化wandb
    if wandb_log:
        wandb.init(
            project="llama2-vicuna-distillation",
            name=f"distill-run-temp{TEMPERATURE}-alpha{ALPHA}-lr{LR}",
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
    
    # 加载学生模型
    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME,
        torch_dtype=torch.bfloat16,  # RTX4090支持bfloat16
        device_map="auto"
    )
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT,
        run_name=f"distill-run-temp{TEMPERATURE}-alpha{ALPHA}-lr{LR}",
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        num_train_epochs=3,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        save_total_limit=2,
        report_to="wandb" if wandb_log else "none",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        learning_rate=LR,
        logging_steps=10,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.01,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        dataloader_num_workers=0,  # 禁用多进程数据加载
        dataloader_pin_memory=True,
        torch_compile=False,  # 暂时禁用编译优化
        gradient_checkpointing=True,
    )
    
    # 初始化训练器
    trainer = DistillationTrainer(
        student,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        temperature=TEMPERATURE,
        alpha=ALPHA,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存最终模型
    print("Saving model...")
    trainer.save_model(MODEL_OUTPUT)
    tokenizer.save_pretrained(MODEL_OUTPUT)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
