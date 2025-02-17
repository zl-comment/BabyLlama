from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import wandb
import json
from peft import LoraConfig, get_peft_model  # 不调用 prepare_model_for_kbit_training
import datetime
import os
import signal
import sys

################### 超参数设置 ###################
LR = 1e-5                           # 学习率
BATCH_SIZE = 18                     # 小批次大小
GRAD_ACCUMULATION = 8               # 梯度累积步数
SEQ_LENGTH = 128                    # 序列长度
TEMPERATURE = 1.0                   # 温度参数
ALPHA = 0.2                         # 蒸馏权重（用于 logits 对齐）
BETA = 0.2                          # 注意力对齐损失的权重

# 是否使用8-bit/4-bit量化（本例均不使用）
USE_8BIT = False
USE_4BIT = False

# 是否启用梯度检查点（若显存允许，可开启以降低显存占用）
USE_GRADIENT_CHECKPOINTING = True

# 训练参数
NUM_EPOCHS = 30                     # 为测试设置较少轮数，正式时可调整
SAVE_STEPS = 500
EVAL_STEPS = 500
WARMUP_STEPS = 2000                 # 延长预热
MAX_GRAD_NORM = 1.0                 # 梯度裁剪阈值

# 模型路径（请根据自己的环境修改）
PATH = Path("./")
TEACHER_MODEL_NAME = r"D:\ZLCODE\model\Llama-2-7b-chat-hf"   # 教师模型路径
STUDENT_MODEL_NAME = r"D:\ZLCODE\model\vicuna-7b-v1.5"         # 学生模型路径
MODEL_OUTPUT = PATH / 'models/vicuna-7b-distilled'

# Wandb配置
wandb_log = True                    # 是否使用 wandb 记录训练日志

# 控制教师模型是否加载到 CPU，显存不足时请设置为 True
LOAD_TEACHER_ON_CPU = True
####################################################


def get_generation_config():
    """获取生成配置"""
    return GenerationConfig(
        do_sample=True,
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
    def __init__(self, targets, goals, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 组合 target 和 goal 成对话格式
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
        return item


class SubsetDataset(Dataset):
    """数据集子集"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def custom_collate_fn(batch):
    """自定义数据整理函数"""
    batch_dict = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        for key in batch_dict:
            if key in item:
                batch_dict[key].append(item[key])
    for key in batch_dict:
        if batch_dict[key]:
            batch_dict[key] = torch.stack(batch_dict[key])
    return batch_dict


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, beta=0.2, **kwargs):
        # 关于 tokenizer 参数的警告可忽略
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError("teacher_model must be provided for attention distillation.")
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        print(f"DistillationTrainer initialized with temperature={temperature}, alpha={alpha}, beta={beta}")
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 获取 labels 并确保在学生模型设备上
        labels = inputs["labels"].to(model.device)

        # 将输入数据迁移到教师模型所在设备（本例教师模型在 CPU 上）
        inputs_for_teacher = {k: v.to(self.teacher_model.device)
                              for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
        inputs_for_teacher["output_attentions"] = True

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs_for_teacher)
        teacher_logits = teacher_outputs.logits  # teacher_logits 在 CPU 上
        teacher_attentions = teacher_outputs.attentions

        # 学生模型前向传播（学生模型在 GPU 上）
        inputs["output_attentions"] = True
        outputs = model(**inputs)
        student_logits = outputs.logits  # 学生模型输出在 GPU 上
        student_attentions = outputs.attentions

        # 将教师输出转移到学生所在设备（GPU），确保数据类型匹配
        shifted_teacher_logits = teacher_logits[:, :-1, :].to(
            student_logits.device, dtype=student_logits.dtype
        ).contiguous()
        shifted_student_logits = student_logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # 计算语言建模损失
        lm_loss = F.cross_entropy(
            shifted_student_logits.view(-1, shifted_student_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100
        )
        # 蒸馏损失：计算教师与学生在 softened logits 下的 KL 散度
        teacher_probs = F.softmax(shifted_teacher_logits / self.temperature + 1e-8, dim=-1)
        student_log_probs = F.log_softmax(shifted_student_logits / self.temperature + 1e-8, dim=-1)
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2)

        # 注意力对齐：取最后 min_layers 层对齐；如果注意力头数不同，则对教师注意力取平均
        min_layers = min(len(student_attentions), len(teacher_attentions))
        aligned_student_attentions = student_attentions[-min_layers:]
        aligned_teacher_attentions = teacher_attentions[-min_layers:]
        attention_loss = 0.0
        for s_att, t_att in zip(aligned_student_attentions, aligned_teacher_attentions):
            if s_att.shape[1] != t_att.shape[1]:
                t_att = t_att.mean(dim=1, keepdim=True)
            # 将教师的注意力转移到学生模型所在设备（GPU）
            attention_loss += F.mse_loss(s_att.float(), t_att.float().to(s_att.device))
        attention_loss = attention_loss / min_layers

        total_loss = (1 - self.alpha - self.beta) * lm_loss + self.alpha * distill_loss + self.beta * attention_loss

        # ★ 确保 loss 与实际需要梯度的参数建立依赖
        dummy_param = None
        for p in model.parameters():
            if p.requires_grad:
                dummy_param = p
                break
        if dummy_param is None:
            raise ValueError("No trainable parameter found!")
        total_loss = total_loss + 0 * dummy_param.sum()

        if self.state.global_step % 10 == 0:
            self.log({
                "train/lm_loss": lm_loss.item(),
                "train/distill_loss": distill_loss.item(),
                "train/attn_loss": attention_loss.item(),
                "train/total_loss": total_loss.item(),
                "train/temperature": self.temperature,
                "train/alpha": self.alpha,
                "train/beta": self.beta
            })

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        total_kl_div = 0
        total_js_div = 0
        total_top1_acc = 0
        total_top5_acc = 0
        total_attn_mse = 0
        n_samples = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                inputs_for_teacher = {k: v.to(self.teacher_model.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
                inputs_for_teacher["output_attentions"] = True
                teacher_outputs = self.teacher_model(**inputs_for_teacher)
                teacher_logits = teacher_outputs.logits
                teacher_attentions = teacher_outputs.attentions
                outputs = self.model(**batch, output_attentions=True)
                student_logits = outputs.logits
                student_attentions = outputs.attentions

                shifted_student_logits = student_logits[:, :-1, :].contiguous()
                shifted_teacher_logits = teacher_logits[:, :-1, :].to(student_logits.dtype).contiguous()
                student_probs = F.log_softmax(shifted_student_logits / self.temperature + 1e-8, dim=-1)
                teacher_probs = F.softmax(shifted_teacher_logits / self.temperature + 1e-8, dim=-1)
                kl_div = F.kl_div(
                    student_probs,
                    teacher_probs,
                    reduction='batchmean',
                    log_target=False
                ) * (self.temperature ** 2)
                student_probs_raw = F.softmax(shifted_student_logits, dim=-1)
                teacher_probs_raw = F.softmax(shifted_teacher_logits, dim=-1)
                m = 0.5 * (student_probs_raw + teacher_probs_raw)
                js_div = 0.5 * (
                    F.kl_div(torch.log(student_probs_raw + 1e-8), m, reduction='batchmean') +
                    F.kl_div(torch.log(teacher_probs_raw + 1e-8), m, reduction='batchmean')
                )
                student_preds = shifted_student_logits.argmax(dim=-1)
                teacher_preds = shifted_teacher_logits.argmax(dim=-1)
                top1_acc = (student_preds == teacher_preds).float().mean()
                _, student_top5 = shifted_student_logits.topk(5, dim=-1)
                top5_acc = torch.any(student_top5 == teacher_preds.unsqueeze(-1), dim=-1).float().mean()
                attn_loss = 0.0
                min_layers = min(len(student_attentions), len(teacher_attentions))
                aligned_student_attentions = student_attentions[-min_layers:]
                aligned_teacher_attentions = teacher_attentions[-min_layers:]
                for s_att, t_att in zip(aligned_student_attentions, aligned_teacher_attentions):
                    if s_att.shape[1] != t_att.shape[1]:
                        t_att = t_att.mean(dim=1, keepdim=True)
                    attn_loss += F.mse_loss(s_att.float(), t_att.float())
                attn_loss = attn_loss / min_layers

                batch_size = batch["input_ids"].size(0)
                total_kl_div += kl_div.item() * batch_size
                total_js_div += js_div.item() * batch_size
                total_top1_acc += top1_acc.item() * batch_size
                total_top5_acc += top5_acc.item() * batch_size
                total_attn_mse += attn_loss.item() * batch_size
                n_samples += batch_size

        if n_samples > 0:
            metrics.update({
                f"{metric_key_prefix}/kl_divergence": total_kl_div / n_samples,
                f"{metric_key_prefix}/js_divergence": total_js_div / n_samples,
                f"{metric_key_prefix}/top1_accuracy": total_top1_acc / n_samples,
                f"{metric_key_prefix}/top5_accuracy": total_top5_acc / n_samples,
                f"{metric_key_prefix}/attn_mse": total_attn_mse / n_samples
            })
        self.log(metrics)
        return metrics

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()
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
        inputs = super()._prepare_inputs(data)
        return inputs


def save_model_on_exit(student_model, tokenizer, versioned_model_path=None):
    """在程序退出时保存模型的函数"""
    if versioned_model_path is None:
        # 如果没有指定保存路径，创建一个新的路径
        final_model_path = MODEL_OUTPUT / "emergency_saved_model"
        i = 1
        while True:
            if i == 1:
                versioned_model_path = final_model_path
            else:
                versioned_model_path = final_model_path.parent / f"emergency_saved_model-{i}"
            if not versioned_model_path.exists():
                break
            i += 1
    
    print("\n\n=== 正在进行紧急保存... ===")
    print(f"保存路径: {versioned_model_path}")
    versioned_model_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("1. 尝试保存 LoRA 适配器...")
        student_model.save_pretrained(versioned_model_path / "lora_adapter")
        print("- LoRA适配器保存成功")
    except Exception as e:
        print(f"保存 LoRA适配器时出错: {str(e)}")
    
    try:
        print("2. 尝试保存分词器...")
        tokenizer.save_pretrained(versioned_model_path)
        print("- 分词器保存成功")
    except Exception as e:
        print(f"保存分词器时出错: {str(e)}")
    
    try:
        print("3. 尝试保存训练配置...")
        training_config = {
            "emergency_save": True,
            "save_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "teacher_model": TEACHER_MODEL_NAME,
            "student_model": STUDENT_MODEL_NAME,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "beta": BETA,
        }
        with open(versioned_model_path / "emergency_training_config.json", "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=4, ensure_ascii=False)
        print("- 训练配置保存成功")
    except Exception as e:
        print(f"保存训练配置时出错: {str(e)}")
    
    print("=== 紧急保存完成 ===\n")


def signal_handler(signum, frame):
    """信号处理函数"""
    print('\n\n捕获到中断信号，正在保存模型...')
    if 'student_model' in globals() and 'tokenizer' in globals():
        save_model_on_exit(student_model, tokenizer)
    print('模型已保存，正在退出程序...')
    sys.exit(0)


def compute_distillation_metrics_separate(student_model, teacher_model, eval_dataset, tokenizer):
    """
    分阶段评估：
      1. 先用教师模型在 GPU 上对整个 eval_dataset 进行推理，
         将输出（logits 和 attentions）转移到 CPU 上保存。
      2. 然后卸载教师模型（或至少释放 GPU 占用），再用学生模型在 GPU 上对整个 eval_dataset 进行推理，
         同样将输出转移到 CPU 上保存。
      3. 最后根据保存的结果计算 KL 散度、交叉熵、预测准确率和注意力对齐损失，并返回各项指标。
    这种方法避免了在同一时间同时加载教师和学生模型在 GPU 上进行推理。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 第一步：教师模型推理 ===
    print("开始使用教师模型推理评估数据集...")
    teacher_model.to(device)
    teacher_model.eval()
    teacher_results = []
    with torch.no_grad():
        for item in eval_dataset:
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
            inputs["output_attentions"] = True
            out = teacher_model(**inputs)
            # 将输出转移到 CPU 保存
            teacher_results.append({
                "logits": out.logits.cpu(),
                "attentions": [att.cpu() for att in out.attentions]
            })
    # 如果需要释放教师模型占用，可以调用下面代码
    teacher_model.cpu()
    torch.cuda.empty_cache()

    # === 第二步：学生模型推理 ===
    print("开始使用学生模型推理评估数据集...")
    student_model.to(device)
    student_model.eval()
    student_results = []
    with torch.no_grad():
        for item in eval_dataset:
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
            inputs["output_attentions"] = True
            out = student_model(**inputs)
            student_results.append({
                "logits": out.logits.cpu(),
                "attentions": [att.cpu() for att in out.attentions]
            })

    # === 第三步：逐样本计算指标 ===
    total_kl_div = 0.0
    total_ce_loss = 0.0
    total_accuracy = 0.0
    total_attn_mse = 0.0
    n_samples = len(eval_dataset)

    for teacher_out, student_out, item in zip(teacher_results, student_results, eval_dataset):
        # teacher_logits 和 student_logits 的形状均为 (1, seq_length, vocab_size)
        teacher_logits = teacher_out["logits"]
        student_logits = student_out["logits"]
        teacher_attentions = teacher_out["attentions"]
        student_attentions = student_out["attentions"]

        # shift 操作：去掉最后一个时间步；同时构造 labels：去掉第一个 token
        shifted_teacher_logits = teacher_logits[:, :-1, :]
        shifted_student_logits = student_logits[:, :-1, :]
        shifted_labels = item["input_ids"].unsqueeze(0)[:, 1:]

        teacher_probs = F.softmax(shifted_teacher_logits / TEMPERATURE + 1e-8, dim=-1)
        student_log_probs = F.log_softmax(shifted_student_logits / TEMPERATURE + 1e-8, dim=-1)
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (TEMPERATURE ** 2)
        ce_loss = F.cross_entropy(shifted_student_logits.view(-1, shifted_student_logits.size(-1)),
                                  shifted_labels.view(-1),
                                  ignore_index=-100)
        student_preds = shifted_student_logits.argmax(dim=-1)
        teacher_preds = shifted_teacher_logits.argmax(dim=-1)
        accuracy = (student_preds == teacher_preds).float().mean()

        # 注意力对齐损失计算：取最后 min_layers 层
        min_layers = min(len(student_attentions), len(teacher_attentions))
        attn_loss = 0.0
        for s_att, t_att in zip(student_attentions[-min_layers:], teacher_attentions[-min_layers:]):
            # 如果注意力头数不一致，可以先对教师注意力取平均
            if s_att.shape[1] != t_att.shape[1]:
                t_att = t_att.mean(dim=1, keepdim=True)
            attn_loss += F.mse_loss(s_att.float(), t_att.float())
        attn_loss = attn_loss / min_layers

        total_kl_div += kl_div.item()
        total_ce_loss += ce_loss.item()
        total_accuracy += accuracy.item()
        total_attn_mse += attn_loss.item()

    metrics = {
        "kl_divergence": total_kl_div / n_samples,
        "cross_entropy_loss": total_ce_loss / n_samples,
        "prediction_accuracy": total_accuracy / n_samples,
        "attn_mse": total_attn_mse / n_samples,
    }
    return metrics


def ensure_dummy_dataset():
    """
    若数据集文件不存在，则生成一个简单的虚拟数据集，便于测试代码运行。
    """
    dataset_path = Path("./data/advbench/harmful_behaviors.csv")
    if not dataset_path.exists():
        print("未找到数据集，正在生成虚拟数据集...")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_data = {
            "target": ["这是回答1", "这是回答2", "这是回答3"],
            "goal": ["目标1", "目标2", "目标3"]
        }
        df = pd.DataFrame(dummy_data)
        df.to_csv(dataset_path, index=False)
        print(f"虚拟数据集已生成：{dataset_path}")


def prepare_datasets(tokenizer):
    """准备训练和评估数据集"""
    data_path = Path("./data/advbench/harmful_behaviors.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"数据集文件不存在：{data_path}")
    data = pd.read_csv(data_path)
    targets = data['target'].tolist()
    goals = data['goal'].tolist()
    dataset = CustomDataset(targets, goals, tokenizer, SEQ_LENGTH)
    print(f"总数据集大小: {len(dataset)}")
    first_item = dataset[0]
    print("\n原始数据集第一个样本:")
    print("Keys:", first_item.keys())
    for k, v in first_item.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape:", v.shape)
    train_size = int(0.8 * len(dataset))
    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]
    train_dataset = SubsetDataset(dataset, train_indices)
    eval_dataset = SubsetDataset(dataset, eval_indices)
    print(f"\n训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")
    first_train_item = train_dataset[0]
    print("\n训练集第一个样本:")
    print("Keys:", first_train_item.keys())
    for k, v in first_train_item.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape:", v.shape)
    return train_dataset, eval_dataset


def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    try:
        # 开启异常检测，有助于定位反向传播时的错误
        torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global tokenizer
        # 使用教师模型的 tokenizer（可根据需要自行选择）
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        generation_config = get_generation_config()
        print("\n=== 模型加载信息 ===")
        print(f"模型输出目录: {MODEL_OUTPUT}")
        latest_checkpoint = get_latest_checkpoint()

        # 根据显存情况，强制将教师模型加载到 CPU
        if LOAD_TEACHER_ON_CPU:
            teacher_dtype = torch.float32  # CPU 上建议使用 float32
            teacher_device_map = {"": "cpu"}
        else:
            teacher_dtype = torch.float16
            teacher_device_map = "auto"

        print("加载教师模型（加载到 CPU）...")
        teacher_model = LlamaForCausalLM.from_pretrained(
            TEACHER_MODEL_NAME,
            load_in_8bit=USE_8BIT,
            load_in_4bit=USE_4BIT,
            device_map=teacher_device_map,
            torch_dtype=teacher_dtype,
            generation_config=generation_config
        )
        teacher_model.config.use_cache = False
        teacher_model.config.attn_implementation = "eager"
        teacher_model.eval()

        if latest_checkpoint is not None:
            model_path = os.path.join(str(latest_checkpoint), "complete_model")
            print(f"正在从检查点加载学生模型: {model_path}")
            student_model = LlamaForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=USE_8BIT,
                load_in_4bit=USE_4BIT,
                device_map="auto",  # 学生模型加载到 GPU 上
                torch_dtype=torch.float16,
                generation_config=generation_config
            )
        else:
            print(f"没有找到检查点，从原始模型加载学生模型: {STUDENT_MODEL_NAME}")
            student_model = LlamaForCausalLM.from_pretrained(
                STUDENT_MODEL_NAME,
                load_in_8bit=USE_8BIT,
                load_in_4bit=USE_4BIT,
                device_map="auto",
                torch_dtype=torch.float16,
                generation_config=generation_config
            )
        student_model.config.use_cache = False
        student_model.config.attn_implementation = "eager"
        print("==================\n")

        # 如果不使用8-bit/4-bit量化，不调用 prepare_model_for_kbit_training
        if USE_8BIT or USE_4BIT:
            from peft import prepare_model_for_kbit_training
            student_model = prepare_model_for_kbit_training(student_model)

        if USE_GRADIENT_CHECKPOINTING:
            student_model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()

        run_name = f"distill-llama2-vicuna-t{TEMPERATURE}-a{ALPHA}-b{BETA}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            save_total_limit=3,
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
            lr_scheduler_type="cosine",
            resume_from_checkpoint=str(latest_checkpoint) if latest_checkpoint is not None else None,
        )
        if wandb_log:
            wandb.init(
                project="llama2-vicuna-distillation",
                name=run_name,
                mode="offline",
                config={
                    "teacher_model": TEACHER_MODEL_NAME,
                    "student_model": STUDENT_MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "alpha": ALPHA,
                    "beta": BETA,
                    "learning_rate": LR,
                    "batch_size": BATCH_SIZE,
                    "sequence_length": SEQ_LENGTH,
                    "train_size": None,
                    "eval_size": None,
                }
            )
        # 确保数据集存在，否则生成虚拟数据集
        ensure_dummy_dataset()
        train_dataset, eval_dataset = prepare_datasets(tokenizer)
        if wandb_log:
            wandb.config.update({
                "train_size": len(train_dataset),
                "eval_size": len(eval_dataset)
            }, allow_val_change=True)
        trainer = DistillationTrainer(
            student_model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            temperature=TEMPERATURE,
            alpha=ALPHA,
            beta=BETA,
        )
        print("Starting training...")
        trainer.train()

        print("\n=== 保存模型 ===")
        final_model_path = MODEL_OUTPUT / "final_model"
        print(f"基础保存路径: {final_model_path}")
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
            print("1. 尝试保存完整模型（合并 LoRA 权重）...")
            try:
                print("方法1: 直接保存合并后的模型...")
                merged_model = student_model.merge_and_unload().cpu()  # 移动到CPU后保存
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
                print("尝试方法2: 转换为 float16 后保存...")
                try:
                    merged_model = student_model.merge_and_unload().cpu()
                    merged_model = merged_model.half()
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
                    print("尝试方法3: 使用 state_dict 保存...")
                    try:
                        merged_model = student_model.merge_and_unload().cpu()
                        save_path = versioned_model_path / "complete_model"
                        save_path.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            merged_model.state_dict(),
                            save_path / "pytorch_model.bin"
                        )
                        print("- 模型状态字典已成功保存（方法3）")
                        del merged_model
                        torch.cuda.empty_cache()
                    except Exception as e3:
                        print(f"所有保存方法都失败。最后一个错误: {str(e3)}")
                        print("将只保存 LoRA 权重和配置文件")
        except Exception as e:
            print(f"合并模型时出错: {str(e)}")
            print("继续保存其他文件...")
        print("2. 保存 LoRA 适配器...")
        try:
            student_model.save_pretrained(versioned_model_path / "lora_adapter")
            print("- LoRA适配器保存成功")
        except Exception as e:
            print(f"保存 LoRA适配器时出错: {str(e)}")
        print("3. 保存分词器...")
        try:
            tokenizer.save_pretrained(versioned_model_path)
            print("- 分词器保存成功")
        except Exception as e:
            print(f"保存分词器时出错: {str(e)}")
        print("4. 保存生成配置...")
        try:
            generation_config = get_generation_config()
            generation_config.save_pretrained(versioned_model_path)
            print("- 生成配置保存成功")
        except Exception as e:
            print(f"保存生成配置时出错: {str(e)}")
        print("5. 保存训练配置...")
        try:
            training_config = {
                "teacher_model": TEACHER_MODEL_NAME,
                "student_model": STUDENT_MODEL_NAME,
                "temperature": TEMPERATURE,
                "alpha": ALPHA,
                "beta": BETA,
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
        print("- 完整模型（合并 LoRA 权重）: /complete_model")
        print("- LoRA适配器: /lora_adapter")
        print("- 分词器")
        print("- 生成配置")
        print("- 训练配置")
        print("================\n")
        # print("Loading complete model for comparison...")
        # original_model = LlamaForCausalLM.from_pretrained(
        #     STUDENT_MODEL_NAME,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )
        # print("Evaluating models...")
        #清理显存
        torch.cuda.empty_cache()
    #     eval_results = compute_distillation_metrics_separate(original_model, teacher_model, eval_dataset, tokenizer)
    #     print("\nEvaluation Results:")
    #     print("Original Model:")
    #     print(f"KL Divergence: {eval_results['kl_divergence']:.4f}")
    #     print(f"Cross Entropy Loss: {eval_results['cross_entropy_loss']:.4f}")
    #     print(f"Prediction Accuracy: {eval_results['prediction_accuracy']:.4f}")
    #     print(f"Attention MSE: {eval_results['attn_mse']:.4f}")
    #     with open(MODEL_OUTPUT / "evaluation_results.json", "w") as f:
    #         json.dump(eval_results, f, indent=4)
    #     print("Training completed!")
    except Exception as e:
        print(f"\n程序发生异常: {str(e)}")
        if 'student_model' in locals() and 'tokenizer' in locals():
            save_model_on_exit(student_model, tokenizer)
        raise  # 重新抛出异常，这样可以看到完整的错误堆栈


if __name__ == "__main__":
    main()
