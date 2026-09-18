"""
prompt_pool_01_03.py
===================
出现在 notebook: 01_数据准备.ipynb
第 3 顺位

功能：定义和管理候选提示策略池（Treatment T）。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import random


@dataclass
class PromptStrategy:
    """
    单个提示策略的封装。
    
    Attributes:
        strategy_id (int): 策略编号（0=基准, 1=CoT, ...）
        name (str): 策略名称
        template (str): 提示模板，使用 {query} 作为查询占位符
        description (str): 策略描述
        cost_multiplier (float): 相对基准策略的预估推理成本倍数（用于成本-效用分析）
    """
    strategy_id: int
    name: str
    template: str
    description: str
    cost_multiplier: float = 1.0
    family: str = ""
    references: str = ""
    requires_examples: bool = False
    requires_multi_sample: bool = False
    n_samples: int = 1
    implementation_note: str = ""


class PromptPool:
    """
    提示策略池管理器。
    
    内置7种标准提示策略：
    T0: Zero-shot Direct (基准)
    T1: Zero-shot Chain-of-Thought
    T2: Few-shot In-context Learning
    T3: Role-play
    T4: Step-by-step Decomposition
    T5: Self-consistency Trigger
    T6: Negative Constraint
    """
    
    DEFAULT_FEW_SHOT_EXAMPLES = {
        "default": (
            "Example 1\n"
            "Question: Choose the best answer from the options.\n"
            "Answer: First identify the relevant clue, then return the final option letter.\n\n"
            "Example 2\n"
            "Question: Solve a short reasoning problem.\n"
            "Answer: Work through the necessary steps and end with 'Final answer: ...'."
        ),
        "math_elementary": (
            "Example 1\n"
            "Question: If Anna has 3 apples and buys 4 more, how many apples does she have?\n"
            "Answer: 3 + 4 = 7. Final answer: 7.\n\n"
            "Example 2\n"
            "Question: A box has 5 rows with 6 pencils in each row. How many pencils are there?\n"
            "Answer: 5 * 6 = 30. Final answer: 30."
        ),
        "math_competition": (
            "Example 1\n"
            "Question: Solve for x: 2x + 3 = 11.\n"
            "Answer: 2x = 8, so x = 4. Final answer: 4.\n\n"
            "Example 2\n"
            "Question: What is 1/2 + 1/3?\n"
            "Answer: Use denominator 6: 3/6 + 2/6 = 5/6. Final answer: 5/6."
        ),
        "commonsense": (
            "Example 1\n"
            "Question: Which object is used to write on paper? (A) spoon (B) pencil (C) shoe\n"
            "Answer: A pencil is used for writing. Final answer: B.\n\n"
            "Example 2\n"
            "Question: What do people usually do when they are thirsty? (A) drink water (B) sleep (C) paint\n"
            "Answer: Drinking water addresses thirst. Final answer: A."
        ),
        "logical_reasoning": (
            "Example 1\n"
            "Question: If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?\n"
            "Answer: The relation is transitive, so yes. Final answer: yes.\n\n"
            "Example 2\n"
            "Question: If today is Monday, what day is two days later?\n"
            "Answer: Tuesday is one day later, Wednesday is two days later. Final answer: Wednesday."
        ),
        "knowledge_qa": (
            "Example 1\n"
            "Question: What planet is known as the Red Planet?\n"
            "Answer: Mars is commonly called the Red Planet. Final answer: Mars.\n\n"
            "Example 2\n"
            "Question: Who wrote Hamlet?\n"
            "Answer: Hamlet was written by William Shakespeare. Final answer: William Shakespeare."
        ),
        "multihop_qa": (
            "Example 1\n"
            "Question: The Eiffel Tower is in a city that is the capital of which country?\n"
            "Answer: The Eiffel Tower is in Paris. Paris is the capital of France. Final answer: France.\n\n"
            "Example 2\n"
            "Question: The author of Hamlet was born in which English town?\n"
            "Answer: Hamlet was written by Shakespeare. Shakespeare was born in Stratford-upon-Avon. Final answer: Stratford-upon-Avon."
        ),
        "code_generation": (
            "Example 1\n"
            "Task: Write a function add_one(x) that returns x plus one.\n"
            "Answer:\n"
            "def add_one(x):\n"
            "    return x + 1\n\n"
            "Example 2\n"
            "Task: Write a function is_even(n) that returns True if n is even.\n"
            "Answer:\n"
            "def is_even(n):\n"
            "    return n % 2 == 0"
        ),
        "summarization": (
            "Example 1\n"
            "Article: A city opened a new library downtown. It will host reading programs for children.\n"
            "Summary: The city opened a downtown library with children's reading programs.\n\n"
            "Example 2\n"
            "Article: Researchers found a battery design that charges faster and lasts longer in lab tests.\n"
            "Summary: Researchers reported a faster-charging, longer-lasting battery design."
        ),
        "translation": (
            "Example 1\n"
            "Source: Guten Morgen.\n"
            "Translation: Good morning.\n\n"
            "Example 2\n"
            "Source: Ich habe ein Buch gelesen.\n"
            "Translation: I read a book."
        ),
    }

    DEFAULT_COMMON_ERRORS = {
        "default": "copying an option without checking it, changing the requested output format, and omitting the final answer",
        "math_elementary": "arithmetic slips, using the wrong operation, and forgetting units",
        "math_competition": "algebraic sign errors, invalid simplification, and mishandling fractions or boxed answers",
        "commonsense": "choosing a plausible but unsupported option and ignoring the provided choices",
        "logical_reasoning": "reversing implications, assuming unstated facts, and skipping edge cases",
        "knowledge_qa": "confusing similar entities, dates, or names",
        "multihop_qa": "answering after only one hop and ignoring entity disambiguation",
        "code_generation": "changing the required function signature, missing edge cases, and returning printed output instead of a value",
        "summarization": "adding unsupported facts, omitting the main event, and writing too verbosely",
        "translation": "dropping named entities, numbers, tense, or negation",
    }

    DEFAULT_POOL = [
        PromptStrategy(
            strategy_id=0,
            name="zero_shot_direct",
            template="{query}\n\nProvide the answer directly. End with 'Final answer: ...'.",
            description="Zero-shot Direct: answer directly without intermediate reasoning",
            cost_multiplier=1.0,
            family="direct_answering",
            references="Brown et al., 2020; benchmark zero-shot baselines",
        ),
        PromptStrategy(
            strategy_id=1,
            name="zero_shot_cot",
            template="{query}\n\nLet's think step by step. End with 'Final answer: ...'.",
            description="Zero-shot Chain-of-Thought: step-by-step reasoning",
            cost_multiplier=2.2,
            family="chain_of_thought",
            references="Wei et al., 2022; Kojima et al., 2022",
        ),
        PromptStrategy(
            strategy_id=2,
            name="few_shot_icl",
            template="Here are task-format examples:\n{examples}\n\nNow answer the new query.\n{query}\n\nEnd with 'Final answer: ...'.",
            description="Few-shot In-context Learning: provide task-format examples",
            cost_multiplier=3.0,
            family="in_context_learning",
            references="Brown et al., 2020; Liu et al., 2022; Rubin et al., 2022",
            requires_examples=True,
        ),
        PromptStrategy(
            strategy_id=3,
            name="role_play",
            template="Act as a careful domain expert. Solve the task faithfully and avoid adding unsupported information.\n\n{query}\n\nEnd with 'Final answer: ...'.",
            description="Role-play: solve as a careful domain expert",
            cost_multiplier=1.1,
            family="persona_prompting",
            references="Choi et al., 2024; role/persona prompting studies",
        ),
        PromptStrategy(
            strategy_id=4,
            name="step_decomposition",
            template="{query}\n\nBreak the problem into the smallest necessary subproblems, solve them in order, and then combine them. End with 'Final answer: ...'.",
            description="Step-by-step Decomposition: decompose the task into subproblems",
            cost_multiplier=2.5,
            family="decomposition_prompting",
            references="Zhou et al., 2022; Wang et al., 2023 Plan-and-Solve prompting",
        ),
        PromptStrategy(
            strategy_id=5,
            name="self_consistency",
            template="{query}\n\nSolve this independently. Show concise reasoning and end with 'Final answer: ...'.",
            description="Self-consistency: aggregate five repeated calls (temperature=0; provider-side nondeterminism yields differing outputs)",
            cost_multiplier=4.0,
            family="self_consistency",
            references="Wang et al., 2022",
            requires_multi_sample=True,
            n_samples=5,
            implementation_note="Run five repeated calls at the global temperature=0 setting and majority-vote the extracted final answers; a single-call variant is an ablation.",
        ),
        PromptStrategy(
            strategy_id=6,
            name="error_aware_verification",
            template="{query}\n\nBefore finalizing, check for these common error types: {common_errors}. End with 'Final answer: ...'.",
            description="Error-aware Verification: explicitly check task-specific common errors",
            cost_multiplier=1.3,
            family="verification_prompting",
            references="Madaan et al., 2023; Shinn et al., 2023; prompt robustness literature",
        ),
    ]
    
    def __init__(self, custom_strategies: Optional[List[PromptStrategy]] = None):
        """
        初始化提示策略池。
        
        Args:
            custom_strategies: 自定义策略列表，若提供则覆盖默认策略
        """
        self.strategies = custom_strategies or self.DEFAULT_POOL.copy()
        self._validate_ids()
    
    def _validate_ids(self):
        """验证策略ID的唯一性和连续性。"""
        ids = [s.strategy_id for s in self.strategies]
        if len(ids) != len(set(ids)):
            raise ValueError("策略ID必须唯一")
        if sorted(ids) != list(range(len(ids))):
            raise ValueError("策略ID必须从0开始连续编号")
    
    def get_strategy(self, strategy_id: int) -> PromptStrategy:
        """根据ID获取策略。"""
        for s in self.strategies:
            if s.strategy_id == strategy_id:
                return s
        raise KeyError(f"策略ID {strategy_id} 不存在")
    
    def get_all_strategies(self) -> List[PromptStrategy]:
        """获取所有策略。"""
        return self.strategies.copy()
    
    def format_prompt(self, strategy_id: int, query: str, examples: Optional[str] = None,
                      task_type: Optional[str] = None, common_errors: Optional[str] = None) -> str:
        """
        将查询格式化为特定策略的提示。
        
        Args:
            strategy_id: 策略ID
            query: 原始查询文本
            examples: Few-shot所需的示例文本（仅T2需要）
        
        Returns:
            str: 格式化后的提示
        """
        strategy = self.get_strategy(strategy_id)
        format_kwargs = {"query": query}

        if "{examples}" in strategy.template:
            format_kwargs["examples"] = examples or self.get_default_examples(task_type)

        if "{common_errors}" in strategy.template:
            format_kwargs["common_errors"] = common_errors or self.get_common_errors(task_type)

        return strategy.template.format(**format_kwargs)

    def get_default_examples(self, task_type: Optional[str] = None) -> str:
        """Return small, task-format few-shot examples so T2 is never an empty-template arm."""
        return self.DEFAULT_FEW_SHOT_EXAMPLES.get(task_type or "", self.DEFAULT_FEW_SHOT_EXAMPLES["default"])

    def get_common_errors(self, task_type: Optional[str] = None) -> str:
        """Return task-specific error checks for the verification arm."""
        return self.DEFAULT_COMMON_ERRORS.get(task_type or "", self.DEFAULT_COMMON_ERRORS["default"])
    
    def assign_random(self, n_queries: int, seed: int = 42) -> List[int]:
        """
        为n个查询随机分配提示策略（均匀随机）。
        
        Args:
            n_queries: 查询数量
            seed: 随机种子
        
        Returns:
            List[int]: 每个查询分配的策略ID列表
        """
        random.seed(seed)
        k = len(self.strategies)
        return [random.randint(0, k - 1) for _ in range(n_queries)]
    
    def assign_stratified(self, query_features: List[Dict], seed: int = 42) -> List[int]:
        """
        分层随机化分配：按任务类型分层，每层内随机分配策略。
        
        Args:
            query_features: 每个查询的特征字典列表，需包含 'task_type' 键
            seed: 随机种子
        
        Returns:
            List[int]: 每个查询分配的策略ID列表
        """
        random.seed(seed)
        assignments = []
        
        # 按任务类型分组
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, feat in enumerate(query_features):
            groups[feat.get("task_type", "unknown")].append(idx)
        
        k = len(self.strategies)
        
        # 每层内随机分配
        for task_type, indices in groups.items():
            shuffled = list(range(k)) * (len(indices) // k + 1)
            random.shuffle(shuffled)
            for i, idx in enumerate(indices):
                assignments.append((idx, shuffled[i]))
        
        # 按原始顺序返回
        assignments.sort(key=lambda x: x[0])
        return [aid for _, aid in assignments]
    
    def get_pool_summary(self) -> str:
        """获取策略池的文本摘要。"""
        lines = ["=" * 50, "Prompt Strategy Pool", "=" * 50]
        for s in self.strategies:
            lines.append(f"T{s.strategy_id}: {s.name} (cost={s.cost_multiplier}x)")
            lines.append(f"  {s.description}")
            lines.append("")
        return "\n".join(lines)

    def get_cost_multipliers(self) -> Dict[int, float]:
        """
        获取所有策略的成本倍数映射。
        
        Returns:
            Dict[int, float]: {strategy_id: cost_multiplier}
        """
        return {s.strategy_id: s.cost_multiplier for s in self.strategies}

    @classmethod
    def from_config(cls, config_strategies: List[Dict]) -> 'PromptPool':
        """
        从配置字典加载策略池（支持 cost_multiplier 字段）。
        
        Args:
            config_strategies: config.yaml 中 prompt_strategies 列表
        """
        strategies = []
        for item in config_strategies:
            strategies.append(PromptStrategy(
                strategy_id=item["id"],
                name=item["name"],
                template=item.get("template", "{query}"),
                description=item.get("description", ""),
                cost_multiplier=item.get("cost_multiplier", 1.0),
                family=item.get("family", item.get("name", "")),
                references=item.get("references", ""),
                requires_examples=item.get("requires_examples", False),
                requires_multi_sample=item.get("requires_multi_sample", False),
                n_samples=item.get("n_samples", 1),
                implementation_note=item.get("implementation_note", ""),
            ))
        return cls(custom_strategies=strategies)
