from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import wandb

############### 超参数设置 ###############
LR = 1e-5               # 较小的学习率，因为是大模型微调
BATCH_SIZE = 4          # 由于模型较大，使用小批量
SEQ_LENGTH = 512        # 序列长度
TEMPERATURE = 2.0       # 温度参数
ALPHA = 0.5             # 蒸馏权重
#######################################

# 设置路径
PATH = Path("./")
TEACHER_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
STUDENT_MODEL_NAME = "lmsys/vicuna-7b-v1.5"
MODEL_OUTPUT = PATH / 'models/vicuna-7b-distilled'

wandb_log = True

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, targets, goals, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 组合target和goal成对话格式
        self.conversations = []
        for target, goal in zip(targets, goals):
            # 构建提示模板
            prompt = f"Human: Given the goal: {goal}\nAssistant: {target}"
            self.conversations.append(prompt)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        # 对话编码
        encoding = self.tokenizer(
            self.conversations[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 去除batch维度
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = item['input_ids'].clone()
        
        return item

# 加载数据
data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
targets = data['target'].tolist()
goals = data['goal'].tolist()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 创建数据集
dataset = CustomDataset(targets, goals, tokenizer, SEQ_LENGTH)

# 80-20分割训练集和验证集
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(
    dataset, [train_size, eval_size]
)

# 加载模型
print("Loading teacher model...")
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
teacher.eval()

print("Loading student model...")
student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 定义蒸馏训练参数类
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

# 定义蒸馏训练器类
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # 计算学生模型输出
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # 计算教师模型输出
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # 验证输出大小一致
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # 计算蒸馏损失
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # 返回加权后的总损失
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

# 初始化wandb
if wandb_log:
    wandb.login()
    wandb.init(project='llama2-vicuna-distillation', name='distillation-run-1')

# 配置训练参数
training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    num_train_epochs=3,
    gradient_accumulation_steps=4,  # 增加梯度累积步数以处理大模型
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_total_limit=2,
    report_to="wandb",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=10,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.01,
    alpha=ALPHA,
    temperature=TEMPERATURE,
)

# 初始化训练器
trainer = DistillationTrainer(
    student,
    training_args,
    teacher_model=teacher,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始训练
print("Starting training...")
trainer.train()

# 保存最终模型
print("Saving model...")
trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)

print("Training completed!")
