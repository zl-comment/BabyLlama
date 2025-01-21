from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    AutoConfig
)
import torch
from pathlib import Path
import json
from peft import PeftModel, LoraConfig, get_peft_model
import transformers

def get_generation_config():
    """获取生成配置"""
    return GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=2,
        repetition_penalty=1.1
    )

def load_model(model_path):
    """加载模型和分词器"""
    print(f"Loading model from: {model_path}")
    
    # 1. 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        raise
    
    # 2. 加载生成配置
    try:
        generation_config = get_generation_config()
        print("Generation config loaded.")
    except Exception as e:
        print(f"Error loading generation config: {str(e)}")
        raise
    
    # 3. 加载模型
    try:
        # 从training_config.json获取基础模型路径
        with open(Path(model_path) / "training_config.json", "r") as f:
            config = json.load(f)
        base_model_path = config["student_model"]
        print(f"Loading base model from: {base_model_path}")
        
        # 设置模型加载参数
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "load_in_8bit": True,
            "low_cpu_mem_usage": True,
        }
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        
        # 加载模型状态字典
        state_dict_path = Path(model_path) / "complete_model" / "pytorch_model.bin"
        if state_dict_path.exists():
            print("Loading model state dictionary...")
            state_dict = torch.load(state_dict_path, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("State dict loaded successfully")
        
        model.eval()
        print(f"Model loaded successfully on device: {next(model.parameters()).device}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, max_length=128):
    """生成回复"""
    try:
        # 构建输入格式
        prompt = f"Human: {prompt}\nAssistant:"
        print(f"Tokenizing prompt: {prompt}")
        
        # 使用简单的tokenization
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        
        # 将输入移到正确的设备
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Device: {model.device}")
        print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
        
        # 使用transformers的Text Generation Pipeline
        print("Generating response...")
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
        # 提取Assistant的回复
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:", 1)[1].strip()
        else:
            response = generated_text.split("Human:", 1)[1].strip()
        
        return response
        
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return f"生成回复时发生错误: {str(e)}"

def main():
    # 设置模型路径
    model_path = "./models/vicuna-7b-distilled/final_model-5"
    
    try:
        # 加载模型和分词器
        model, tokenizer = load_model(model_path)
        
        # 测试生成
        print("\n模型加载完成，开始交互...")
        print("输入 'exit' 退出")
        print("输入 'clear' 清除对话历史")
        
        while True:
            try:
                prompt = input("\nUser: ")
                if prompt.lower() == 'exit':
                    break
                elif not prompt.strip():
                    continue
                
                print("\nAssistant: ", end='', flush=True)
                response = generate_response(model, tokenizer, prompt)
                print(response)
                
            except KeyboardInterrupt:
                print("\n收到中断信号，退出程序...")
                break
            except Exception as e:
                print(f"\n处理输入时发生错误: {str(e)}")
                continue
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
