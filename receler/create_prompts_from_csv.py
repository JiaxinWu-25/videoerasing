"""
从评估数据 CSV 文件中提取 prompts 用于训练

使用方法：
    python create_prompts_from_csv.py \
        --csv_file evaluation/data/nudity_cogvideox.csv \
        --output_file prompts.txt \
        --column_name prompt
"""

import argparse
import pandas as pd
from pathlib import Path


def extract_prompts_from_csv(
    csv_file: str,
    output_file: str,
    column_name: str = "prompt",
    max_prompts: int = None
):
    """
    从 CSV 文件中提取 prompts
    
    Args:
        csv_file: CSV 文件路径
        output_file: 输出文件路径
        column_name: 包含 prompt 的列名
        max_prompts: 最大提取数量（None 表示全部）
    """
    # 读取 CSV
    print(f"读取 CSV 文件: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if column_name not in df.columns:
        raise ValueError(f"列 '{column_name}' 不存在于 CSV 文件中。可用列: {df.columns.tolist()}")
    
    # 提取 prompts
    prompts = df[column_name].dropna().tolist()
    
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    
    print(f"提取了 {len(prompts)} 个 prompts")
    
    # 写入文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Prompts extracted from {csv_file}\n")
        f.write(f"# Total: {len(prompts)} prompts\n")
        f.write("# Format: one prompt per line\n\n")
        
        for prompt in prompts:
            # 清理 prompt（移除引号等）
            prompt_clean = str(prompt).strip().strip('"').strip("'")
            if prompt_clean:
                f.write(f"{prompt_clean}\n")
    
    print(f"Prompts 已保存到: {output_file}")
    print(f"  前 3 个 prompts:")
    for i, prompt in enumerate(prompts[:3], 1):
        print(f"    {i}. {prompt[:80]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 CSV 文件提取 prompts")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="CSV 文件路径（如 evaluation/data/nudity_cogvideox.csv）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="prompts.txt",
        help="输出文件路径"
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="prompt",
        help="包含 prompt 的列名"
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="最大提取数量（None 表示全部）"
    )
    
    args = parser.parse_args()
    
    extract_prompts_from_csv(
        csv_file=args.csv_file,
        output_file=args.output_file,
        column_name=args.column_name,
        max_prompts=args.max_prompts
    )

