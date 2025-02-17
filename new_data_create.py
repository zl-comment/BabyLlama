import pandas as pd
import nltk
from nltk.corpus import wordnet
import random
from pathlib import Path

# 下载WordNet资源
nltk.download('wordnet')
nltk.download('omw-1.4')


# 获取同义词
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


# 同义词替换
def synonym_replacement(sentence):
    words = sentence.split()
    new_sentence = []

    for word in words:
        synonyms = get_synonyms(word)
        if len(synonyms) > 1:  # 确保有多个同义词
            synonym = random.choice(synonyms)
            new_sentence.append(synonym if synonym != word else word)
        else:
            new_sentence.append(word)

    return ' '.join(new_sentence)


# 从CSV读取数据并扩展
def expand_dataset(input_csv, output_csv):
    # 读取 CSV 数据
    df = pd.read_csv(input_csv)

    # 假设你要处理的列是 'goal' 和 'target'
    if 'goal' not in df.columns or 'target' not in df.columns:
        print("Error: The input CSV must contain 'goal' and 'target' columns.")
        return

    expanded_goals = []
    expanded_targets = []

    # 对每一行应用同义词替换生成新数据
    for index, row in df.iterrows():
        goal = row['goal']
        target = row['target']

        goal_augmented = [synonym_replacement(goal) for _ in range(4)]
        target_augmented = [synonym_replacement(target) for _ in range(4)]

        expanded_goals.extend(goal_augmented)
        expanded_targets.extend(target_augmented)

    # 将生成的新数据存入新列
    new_data = pd.DataFrame({
        'goal': expanded_goals,
        'target': expanded_targets
    })

    # 保存为新的 CSV 文件
    new_data.to_csv(output_csv, index=False)
    print(f"Augmented dataset saved to {output_csv}")

dataset_path = Path("./data/advbench/harmful_behaviors.csv")

# 调用函数，传入输入和输出文件路径
input_csv = dataset_path  # 你原始的CSV文件
output_csv = './data/advbench/augmented_data.csv'  # 生成的新CSV文件
expand_dataset(input_csv, output_csv)
