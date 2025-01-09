# 导入必要的库和模块
from transformers import (
    GPT2TokenizerFast,  # GPT2分词器
    LlamaForCausalLM,   # Llama因果语言模型
    LlamaConfig,        # Llama配置
    GPT2LMHeadModel,    # GPT2语言模型
    Trainer,            # Huggingface训练器
    TrainingArguments,  # 训练参数
    DataCollatorForLanguageModeling, # 数据整理器
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from random import sample
from pathlib import Path
import wandb  # 用于实验跟踪和可视化

from babylm_dataset import BabylmDataset

############### 超参数设置 ###############
LR = 2.5e-4              # 学习率
BATCH_SIZE = 32          # 批次大小
SEQ_LENGTH = 128         # 序列长度

# 知识蒸馏相关参数
TEMPERATURE = 2.0        # 温度参数，用于软化概率分布
ALPHA = 0.5              # 平衡学生损失和蒸馏损失的权重
#######################################

# 设置路径
PATH = Path("./")
# 教师模型路径
teacher_dir1 = PATH / 'models/Llama-360M'  # 第一个教师模型：Llama-360M
teacher_dir2 = PATH / 'models/gpt-705M'    # 第二个教师模型：GPT-705M

# 学生模型配置
MODEL_NAME = f'Baby-Llama-58M'
MODEL_OUTPUT = Path('./models') / MODEL_NAME
EVAL_SAMPLES = 8192      # 评估样本数量

wandb_log = True        # 是否启用wandb日志

# 配置tokenizer
tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"        # 句子开始标记
tokenizer.eos_token = "</s>"       # 句子结束标记
tokenizer.pad_token = "<pad>"      # 填充标记

# 加载训练和评估数据集
train_dataset = BabylmDataset(PATH / "data/babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

# 随机采样评估数据集
eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

tokenizer.model_max_length = SEQ_LENGTH

# 配置学生模型（Baby Llama）
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,                # 隐藏层大小
    num_hidden_layers=16,           # 隐藏层数量
    intermediate_size=1024,         # 中间层大小
    num_attention_heads=8,          # 注意力头数量
    bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    max_position_embeddings=2*SEQ_LENGTH,
)

# 初始化模型
student = LlamaForCausalLM(config)  # 学生模型
teacher1 = LlamaForCausalLM.from_pretrained(teacher_dir1)  # 教师模型1
teacher2 = GPT2LMHeadModel.from_pretrained(teacher_dir2)   # 教师模型2
teachers = [teacher1, teacher2]

# 设置数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# 打印模型参数量
print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')

# 定义蒸馏训练参数类
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha          # 损失权重
        self.temperature = temperature  # 温度参数

# 定义蒸馏训练器类
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        # 将教师模型移到与学生模型相同的设备上
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # 计算学生模型输出
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # 计算教师模型输出（集成多个教师的预测）
        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # 验证输出大小一致
        assert outputs_student.logits.size() == avg_teacher_logits.size()

        # 计算蒸馏损失（KL散度）
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # 返回加权后的总损失
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

# 初始化wandb日志
if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME)

# 配置训练参数
training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",          # 每个epoch保存一次
    evaluation_strategy="epoch",    # 每个epoch评估一次
    num_train_epochs=6,            # 训练轮数
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,            # 最多保存多少个检查点
    report_to="wandb",             # 使用wandb记录日志
    warmup_steps=200,              # 预热步数
    lr_scheduler_type="cosine",    # 学习率调度器类型
    learning_rate=LR,
    logging_steps=20,
    fp16=True,                     # 使用混合精度训练
    load_best_model_at_end=True,   # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss",
    weight_decay=0.1,              # 权重衰减
    alpha=ALPHA,
    temperature=TEMPERATURE,
)

# 初始化训练器
trainer = DistillationTrainer(
    student,
    training_args,
    teacher_models=teachers,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始训练
trainer.train()

# 保存模型和分词器
trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)
