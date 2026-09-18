# PromptForest Strategy Pool: Complete Template Documentation (T0–T6)

This document gives the complete, verbatim prompt templates for the seven prompt
strategies (arms T0–T6) used in the PromptForest experiments, together with the
exact example-selection and error-list content. The executable reference is
`prompts/prompt_pool_01_03.py` in the same directory; every string below is copied
verbatim from that file. In all templates, `{query}` is replaced by the raw query
text.

## T0 — Zero-shot Direct (baseline)

```text
{query}

Provide the answer directly. End with 'Final answer: ...'.
```

Cost multiplier 1.0.

## T1 — Zero-shot Chain-of-Thought

```text
{query}

Let's think step by step. End with 'Final answer: ...'.
```

Cost multiplier 2.2.

## T2 — Few-shot In-context Learning

```text
Here are task-format examples:
{examples}

Now answer the new query.
{query}

End with 'Final answer: ...'.
```

Cost multiplier 3.0.

**Example selection logic (actual behavior of `PromptPool.get_default_examples`).**
The examples are NOT retrieved, ranked, or sampled: the pool stores a static
dictionary `DEFAULT_FEW_SHOT_EXAMPLES` mapping each of the nine task types to a
fixed, hand-written block of exactly **two** examples, plus a `default` block.
At formatting time the code performs a plain dictionary lookup by the query's
`task_type` (falling back to `default` if the task type is absent or unknown) and
injects the block verbatim, in the fixed written order (Example 1, then Example 2).
The examples are synthetic, task-format illustrations written by the authors —
they are not drawn from any benchmark's training split, so no benchmark data leaks
into the prompt. The two-example blocks are reproduced in full in the Appendix
section below and in `prompt_pool_01_03.py`.

## T3 — Role-play (persona)

```text
Act as a careful domain expert. Solve the task faithfully and avoid adding unsupported information.

{query}

End with 'Final answer: ...'.
```

Cost multiplier 1.1. The persona is a single fixed generic instruction ("careful
domain expert"); it is not specialized per task.

## T4 — Step-by-step Decomposition

```text
{query}

Break the problem into the smallest necessary subproblems, solve them in order, and then combine them. End with 'Final answer: ...'.
```

Cost multiplier 2.5.

## T5 — Self-consistency Trigger

```text
{query}

Solve this independently. Show concise reasoning and end with 'Final answer: ...'.
```

Cost multiplier 4.0. **Multi-sample protocol (as actually executed during data
collection):** this arm is run with `n_samples = 5` repeated calls per query at
the same global setting `temperature = 0.0` used for all arms; the `Final answer:`
field is extracted from each call (regex on `final answer: ...`, falling back to
the last non-empty line) and the arm's answer is the **majority vote** over the
five extracted answers (`aggregation = 'majority_vote_final_answer'` in the
collection code; reported latency is the sum of the five calls). Note that despite temperature 0.0,
provider-side nondeterminism produced differing raw responses in 1083/1065/1042 of
1,100 queries for GPT-5.5 / DeepSeek-v4-Pro / Claude-Sonnet-4.6 (see
`t5_repeat_diversity_check.csv` in this directory), so the vote is over genuinely
different outputs; a single-call variant is treated only as an ablation.

## T6 — Error-aware Verification (negative constraint)

```text
{query}

Before finalizing, check for these common error types: {common_errors}. End with 'Final answer: ...'.
```

Cost multiplier 1.3. `{common_errors}` is filled by `PromptPool.get_common_errors`,
a static per-task-type lookup (fallback: `default`). The complete list, verbatim:

| task_type | common_errors |
|---|---|
| default | copying an option without checking it, changing the requested output format, and omitting the final answer |
| math_elementary | arithmetic slips, using the wrong operation, and forgetting units |
| math_competition | algebraic sign errors, invalid simplification, and mishandling fractions or boxed answers |
| commonsense | choosing a plausible but unsupported option and ignoring the provided choices |
| logical_reasoning | reversing implications, assuming unstated facts, and skipping edge cases |
| knowledge_qa | confusing similar entities, dates, or names |
| multihop_qa | answering after only one hop and ignoring entity disambiguation |
| code_generation | changing the required function signature, missing edge cases, and returning printed output instead of a value |
| summarization | adding unsupported facts, omitting the main event, and writing too verbosely |
| translation | dropping named entities, numbers, tense, or negation |

## Appendix: complete T2 few-shot example blocks (verbatim)

### default

```text
Example 1
Question: Choose the best answer from the options.
Answer: First identify the relevant clue, then return the final option letter.

Example 2
Question: Solve a short reasoning problem.
Answer: Work through the necessary steps and end with 'Final answer: ...'.
```

### math_elementary

```text
Example 1
Question: If Anna has 3 apples and buys 4 more, how many apples does she have?
Answer: 3 + 4 = 7. Final answer: 7.

Example 2
Question: A box has 5 rows with 6 pencils in each row. How many pencils are there?
Answer: 5 * 6 = 30. Final answer: 30.
```

### math_competition

```text
Example 1
Question: Solve for x: 2x + 3 = 11.
Answer: 2x = 8, so x = 4. Final answer: 4.

Example 2
Question: What is 1/2 + 1/3?
Answer: Use denominator 6: 3/6 + 2/6 = 5/6. Final answer: 5/6.
```

### commonsense

```text
Example 1
Question: Which object is used to write on paper? (A) spoon (B) pencil (C) shoe
Answer: A pencil is used for writing. Final answer: B.

Example 2
Question: What do people usually do when they are thirsty? (A) drink water (B) sleep (C) paint
Answer: Drinking water addresses thirst. Final answer: A.
```

### logical_reasoning

```text
Example 1
Question: If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?
Answer: The relation is transitive, so yes. Final answer: yes.

Example 2
Question: If today is Monday, what day is two days later?
Answer: Tuesday is one day later, Wednesday is two days later. Final answer: Wednesday.
```

### knowledge_qa

```text
Example 1
Question: What planet is known as the Red Planet?
Answer: Mars is commonly called the Red Planet. Final answer: Mars.

Example 2
Question: Who wrote Hamlet?
Answer: Hamlet was written by William Shakespeare. Final answer: William Shakespeare.
```

### multihop_qa

```text
Example 1
Question: The Eiffel Tower is in a city that is the capital of which country?
Answer: The Eiffel Tower is in Paris. Paris is the capital of France. Final answer: France.

Example 2
Question: The author of Hamlet was born in which English town?
Answer: Hamlet was written by Shakespeare. Shakespeare was born in Stratford-upon-Avon. Final answer: Stratford-upon-Avon.
```

### code_generation

```text
Example 1
Task: Write a function add_one(x) that returns x plus one.
Answer:
def add_one(x):
    return x + 1

Example 2
Task: Write a function is_even(n) that returns True if n is even.
Answer:
def is_even(n):
    return n % 2 == 0
```

### summarization

```text
Example 1
Article: A city opened a new library downtown. It will host reading programs for children.
Summary: The city opened a downtown library with children's reading programs.

Example 2
Article: Researchers found a battery design that charges faster and lasts longer in lab tests.
Summary: Researchers reported a faster-charging, longer-lasting battery design.
```

### translation

```text
Example 1
Source: Guten Morgen.
Translation: Good morning.

Example 2
Source: Ich habe ein Buch gelesen.
Translation: I read a book.
```

## Strategy assignment during data collection

The outcome matrix is full-factorial: every query was evaluated under all seven
strategies (7,700 rows per model). The `assign_stratified` helper in
`prompt_pool_01_03.py` (grouping by `task_type` and assigning shuffled arms) is
retained for prospective partial-feedback settings only and was **not** used to
construct this matrix.
