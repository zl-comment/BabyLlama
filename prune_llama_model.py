import torch
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os
import json

class LlamaModelPruner:
    def __init__(self, model_path, dataset_path, target_size_billions=3.5):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.target_size = target_size_billions
        
        # Load model and tokenizer
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        
        # Calculate target parameters
        self.current_params = sum(p.numel() for p in self.model.parameters())
        self.target_params = int(target_size_billions * 1e9)
        
        # Load and prepare dataset
        self.load_dataset()
        
    def load_dataset(self):
        df = pd.read_csv(self.dataset_path)
        self.examples = df['goal'].tolist()
        
    def get_attention_head_importance(self, num_samples=100):
        importance_scores = {}
        
        # Take a subset of examples for evaluation
        eval_examples = self.examples[:num_samples]
        
        for layer_idx in tqdm(range(len(self.model.model.layers))):
            layer = self.model.model.layers[layer_idx]
            num_heads = layer.self_attn.num_heads
            
            for head_idx in range(num_heads):
                total_similarity = 0
                
                # Evaluate each head's importance
                for example in eval_examples:
                    inputs = self.tokenizer(example, return_tensors="pt").to(self.model.device)
                    
                    # Get original output embedding
                    with torch.no_grad():
                        original_output = self.model(**inputs).logits
                        
                    # Temporarily mask the head
                    attn_mask = torch.ones(num_heads).to(self.model.device)
                    attn_mask[head_idx] = 0
                    
                    # Store original attention weights
                    original_weights = layer.self_attn.q_proj.weight.data.clone()
                    
                    # Zero out the head's weights
                    head_size = layer.self_attn.head_dim
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    layer.self_attn.q_proj.weight.data[:, start_idx:end_idx] *= 0
                    
                    # Get masked output embedding
                    with torch.no_grad():
                        masked_output = self.model(**inputs).logits
                    
                    # Restore original weights
                    layer.self_attn.q_proj.weight.data.copy_(original_weights)
                    
                    # Calculate similarity between original and masked outputs
                    similarity = cosine_similarity(
                        original_output.cpu().numpy().reshape(1, -1),
                        masked_output.cpu().numpy().reshape(1, -1)
                    )[0][0]
                    
                    total_similarity += similarity
                
                # Average similarity across examples
                avg_similarity = total_similarity / len(eval_examples)
                importance_scores[f"layer_{layer_idx}_head_{head_idx}"] = 1 - avg_similarity
        
        return importance_scores
    
    def prune_model(self, importance_scores):
        # Sort heads by importance
        sorted_heads = sorted(importance_scores.items(), key=lambda x: x[1])
        
        # Calculate number of heads to prune
        params_to_remove = self.current_params - self.target_params
        params_per_head = self.model.config.hidden_size * self.model.config.hidden_size // self.model.config.num_attention_heads
        heads_to_prune = int(params_to_remove / params_per_head)
        
        print(f"Pruning {heads_to_prune} attention heads...")
        
        # Prune least important heads
        heads_to_remove = sorted_heads[:heads_to_prune]
        
        # Create pruning configuration
        pruning_config = {
            "pruned_heads": {},
            "original_num_params": self.current_params,
            "target_num_params": self.target_params,
            "pruned_heads_list": [head[0] for head in heads_to_remove]
        }
        
        # Actually prune the heads
        for head_name, _ in heads_to_remove:
            layer_idx = int(head_name.split("_")[1])
            head_idx = int(head_name.split("_")[3])
            
            if layer_idx not in pruning_config["pruned_heads"]:
                pruning_config["pruned_heads"][layer_idx] = []
            
            pruning_config["pruned_heads"][layer_idx].append(head_idx)
            
            # Zero out the head's weights
            layer = self.model.model.layers[layer_idx]
            head_size = layer.self_attn.head_dim
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            
            with torch.no_grad():
                layer.self_attn.q_proj.weight.data[:, start_idx:end_idx] *= 0
                layer.self_attn.k_proj.weight.data[:, start_idx:end_idx] *= 0
                layer.self_attn.v_proj.weight.data[:, start_idx:end_idx] *= 0
        
        return pruning_config
    
    def save_pruned_model(self, output_dir, pruning_config):
        # Save the pruned model
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save pruning configuration
        with open(os.path.join(output_dir, "pruning_config.json"), "w") as f:
            json.dump(pruning_config, f, indent=2)
        
        print(f"Pruned model saved to {output_dir}")

def main():
    # Configuration
    model_path = "D:/ZLCODE/model/Llama-2-7b-chat-hf"
    dataset_path = "./data/advbench/harmful_behaviors.csv"  # Path to your dataset
    output_dir = "pruned_llama_3.5b"
    target_size_billions = 3.5
    
    # Initialize pruner
    pruner = LlamaModelPruner(model_path, dataset_path, target_size_billions)
    
    # Get attention head importance scores
    print("Calculating attention head importance...")
    importance_scores = pruner.get_attention_head_importance()
    
    # Prune model
    print("Pruning model...")
    pruning_config = pruner.prune_model(importance_scores)
    
    # Save pruned model
    print("Saving pruned model...")
    pruner.save_pruned_model(output_dir, pruning_config)
    
    print("Model pruning completed!")

if __name__ == "__main__":
    main()
