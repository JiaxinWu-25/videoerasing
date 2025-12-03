"""
自动生成 Preserved Concepts（保留概念）

根据目标概念自动生成语义相关的保留概念列表，用于防止去学习导致的灾难性遗忘。
"""

import os
import json
from typing import List, Dict, Set
from pathlib import Path


class PreservedConceptsGenerator:
    """
    自动生成保留概念的生成器
    
    使用多种策略：
    1. 预定义的概念映射（最可靠）
    2. LLM生成（如果可用）
    3. 基于语义相似度的生成（可选）
    """
    
    def __init__(self):
        # 预定义的概念映射字典
        # 格式：{目标概念: [相关保留概念列表]}
        self.concept_mapping = {
            # Nudity 相关
            "nudity": [
                "person", "human", "people", "individual",
                "face", "facial features", "expression",
                "clothing", "garment", "outfit", "dress",
                "background", "scene", "setting", "environment",
                "body", "posture", "gesture", "movement",
                "hair", "skin", "hands", "legs"
            ],
            "naked": [
                "person", "human", "people",
                "face", "facial features",
                "clothing", "garment",
                "background", "scene",
                "body", "posture", "gesture",
                "hair", "hands", "legs"
            ],
            "explicit": [
                "person", "human", "people",
                "face", "expression",
                "clothing", "garment",
                "background", "scene", "setting",
                "body", "posture", "movement"
            ],
            "adult content": [
                "person", "human", "people",
                "face", "facial features",
                "clothing", "garment",
                "background", "scene",
                "body", "posture"
            ],
            
            # Face/Person 相关
            "face": [
                "person", "human", "people", "individual",
                "body", "posture", "gesture",
                "clothing", "garment", "outfit",
                "background", "scene", "setting",
                "hair", "hands", "expression",
                "eyes", "nose", "mouth"
            ],
            "person": [
                "face", "facial features", "expression",
                "clothing", "garment", "outfit",
                "background", "scene", "setting", "environment",
                "body", "posture", "gesture", "movement",
                "hair", "hands", "legs", "arms"
            ],
            "angela merkel": [
                "person", "human", "people",
                "face", "facial features", "expression",
                "clothing", "suit", "formal wear", "garment",
                "background", "scene", "setting", "environment",
                "body", "posture", "gesture", "movement",
                "hair", "hands", "speaking", "walking"
            ],
            
            # Object 相关
            "airplane": [
                "sky", "clouds", "air", "atmosphere",
                "background", "scene", "landscape",
                "airport", "runway", "building",
                "person", "people", "pilot",
                "ground", "trees", "mountains"
            ],
            "car": [
                "road", "street", "highway",
                "background", "scene", "landscape",
                "person", "people", "driver",
                "building", "trees", "sky",
                "ground", "parking", "traffic"
            ],
            "bicycle": [
                "person", "people", "rider",
                "road", "street", "path",
                "background", "scene", "landscape",
                "clothing", "helmet", "garment",
                "sky", "trees", "ground"
            ],
            
            # Violence 相关
            "violence": [
                "person", "human", "people",
                "face", "expression",
                "clothing", "garment",
                "background", "scene", "setting",
                "body", "posture", "movement",
                "environment", "location"
            ],
            "weapon": [
                "person", "human", "people",
                "face", "expression",
                "clothing", "garment",
                "background", "scene", "setting",
                "body", "posture", "hands",
                "environment", "location"
            ],
            
            # 通用概念
            "object": [
                "background", "scene", "setting", "environment",
                "person", "people", "human",
                "surface", "table", "ground",
                "lighting", "shadow", "color"
            ],
            "animal": [
                "background", "scene", "setting", "environment",
                "person", "people", "human",
                "nature", "trees", "grass", "sky",
                "habitat", "cage", "enclosure"
            ]
        }
        
        # 通用保留概念（适用于所有目标概念）
        self.common_preserved_concepts = [
            "background", "scene", "setting", "environment",
            "lighting", "color", "texture",
            "composition", "framing", "camera angle"
        ]
    
    def normalize_concept(self, concept: str) -> str:
        """
        标准化概念名称（用于匹配）
        
        Args:
            concept: 原始概念
        
        Returns:
            标准化后的概念
        """
        concept = concept.lower().strip()
        # 移除常见修饰词
        concept = concept.replace("the ", "").replace("a ", "").replace("an ", "")
        return concept
    
    def find_matching_concepts(self, target_concept: str) -> List[str]:
        """
        在预定义映射中查找匹配的概念
        
        Args:
            target_concept: 目标概念
        
        Returns:
            匹配的保留概念列表
        """
        normalized = self.normalize_concept(target_concept)
        
        # 精确匹配
        if normalized in self.concept_mapping:
            return self.concept_mapping[normalized]
        
        # 部分匹配（检查目标概念是否包含映射中的键）
        for key, preserved in self.concept_mapping.items():
            if key in normalized or normalized in key:
                return preserved
        
        # 检查目标概念是否包含常见关键词
        if "nudity" in normalized or "naked" in normalized or "nude" in normalized:
            return self.concept_mapping.get("nudity", [])
        elif "face" in normalized or "person" in normalized or "human" in normalized:
            return self.concept_mapping.get("person", [])
        elif "violence" in normalized or "weapon" in normalized:
            return self.concept_mapping.get("violence", [])
        elif "airplane" in normalized or "aircraft" in normalized or "plane" in normalized:
            return self.concept_mapping.get("airplane", [])
        elif "car" in normalized or "vehicle" in normalized or "automobile" in normalized:
            return self.concept_mapping.get("car", [])
        elif "bicycle" in normalized or "bike" in normalized:
            return self.concept_mapping.get("bicycle", [])
        
        return []
    
    def generate_with_llm(self, target_concept: str, num_concepts: int = 15) -> List[str]:
        """
        使用LLM生成保留概念（可选功能）
        
        Args:
            target_concept: 目标概念
            num_concepts: 生成的概念数量
        
        Returns:
            生成的保留概念列表
        """
        # 这里可以集成LLM API或本地模型
        # 示例：使用简单的提示模板
        
        prompt = f"""Given a target concept to be erased: "{target_concept}"

Generate {num_concepts} semantically related concepts that should be PRESERVED during unlearning to prevent catastrophic forgetting. These concepts should be:
1. Related to the target concept but NOT the same
2. Commonly co-occur with the target concept
3. Important for maintaining model capabilities

Examples:
- If erasing "nudity", preserve: person, face, clothing, background, scene
- If erasing "airplane", preserve: sky, clouds, airport, person, ground

Generate concepts (one per line, simple words or short phrases):"""

        # TODO: 实际使用时可以调用LLM API
        # 这里返回空列表，表示未使用LLM
        return []
    
    def generate(
        self,
        target_concept: str,
        num_concepts: int = 15,
        use_llm: bool = False,
        include_common: bool = True
    ) -> List[str]:
        """
        生成保留概念列表
        
        Args:
            target_concept: 目标概念（要消除的）
            num_concepts: 期望的概念数量（约）
            use_llm: 是否使用LLM生成（如果可用）
            include_common: 是否包含通用保留概念
        
        Returns:
            保留概念列表
        """
        preserved_concepts = set()
        
        # 1. 从预定义映射中查找
        mapped_concepts = self.find_matching_concepts(target_concept)
        preserved_concepts.update(mapped_concepts)
        
        # 2. 使用LLM生成（如果启用且可用）
        if use_llm:
            llm_concepts = self.generate_with_llm(target_concept, num_concepts)
            preserved_concepts.update(llm_concepts)
        
        # 3. 添加通用保留概念
        if include_common:
            preserved_concepts.update(self.common_preserved_concepts)
        
        # 4. 去重并排序
        preserved_list = sorted(list(preserved_concepts))
        
        # 5. 调整数量到目标范围
        if len(preserved_list) > num_concepts:
            # 优先保留更相关的概念（映射中的概念）
            mapped_set = set(mapped_concepts)
            priority_concepts = [c for c in preserved_list if c in mapped_set]
            other_concepts = [c for c in preserved_list if c not in mapped_set]
            
            # 保留所有映射的概念，然后添加其他概念直到达到目标数量
            result = priority_concepts[:]
            remaining = num_concepts - len(result)
            if remaining > 0:
                result.extend(other_concepts[:remaining])
            preserved_list = result
        elif len(preserved_list) < num_concepts and len(mapped_concepts) > 0:
            # 如果概念太少，可以重复映射的概念
            while len(preserved_list) < num_concepts and len(mapped_concepts) > 0:
                preserved_list.extend(mapped_concepts)
            preserved_list = preserved_list[:num_concepts]
        
        return preserved_list[:num_concepts] if len(preserved_list) > num_concepts else preserved_list
    
    def save_to_file(
        self,
        target_concept: str,
        output_file: str,
        num_concepts: int = 15,
        use_llm: bool = False,
        include_common: bool = True
    ):
        """
        生成并保存保留概念到文件
        
        Args:
            target_concept: 目标概念
            output_file: 输出文件路径
            num_concepts: 期望的概念数量
            use_llm: 是否使用LLM
            include_common: 是否包含通用概念
        """
        preserved_concepts = self.generate(
            target_concept=target_concept,
            num_concepts=num_concepts,
            use_llm=use_llm,
            include_common=include_common
        )
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Preserved Concepts for Target Concept: {target_concept}\n")
            f.write(f"# Auto-generated preserved concepts\n")
            f.write(f"# Total: {len(preserved_concepts)} concepts\n")
            f.write("# Format: one concept per line\n\n")
            
            for concept in preserved_concepts:
                f.write(f"{concept}\n")
        
        print(f"✓ 已生成 {len(preserved_concepts)} 个保留概念")
        print(f"✓ 保存到: {output_file}")
        print(f"\n保留概念列表:")
        for i, concept in enumerate(preserved_concepts, 1):
            print(f"  {i:2d}. {concept}")


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="自动生成 Preserved Concepts（保留概念）"
    )
    parser.add_argument(
        "--target_concept",
        type=str,
        required=True,
        help="目标概念（要消除的，如 'nudity', 'airplane'）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出文件路径（默认：preserved_concepts_{target_concept}.txt）"
    )
    parser.add_argument(
        "--num_concepts",
        type=int,
        default=15,
        help="期望的保留概念数量（默认：15）"
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="使用LLM生成（如果可用）"
    )
    parser.add_argument(
        "--no_common",
        action="store_true",
        help="不包含通用保留概念"
    )
    
    args = parser.parse_args()
    
    # 生成输出文件名
    if args.output_file is None:
        safe_name = args.target_concept.lower().replace(" ", "_").replace("/", "_")
        args.output_file = f"preserved_concepts_{safe_name}.txt"
    
    # 生成保留概念
    generator = PreservedConceptsGenerator()
    generator.save_to_file(
        target_concept=args.target_concept,
        output_file=args.output_file,
        num_concepts=args.num_concepts,
        use_llm=args.use_llm,
        include_common=not args.no_common
    )


if __name__ == "__main__":
    main()

