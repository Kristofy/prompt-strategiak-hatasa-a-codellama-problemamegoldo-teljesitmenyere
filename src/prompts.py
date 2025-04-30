PROMPT_SIMPLE = """\
You will be given a mathematical problem and you are expected to provide a working python code that solves the problem. The code should be correct and efficient.

The problem is as follows:
{problem}

Only your last python code will be evaluated, you are required to provide the code blocks in markdown format using triple backticks and the word 'python' (i.e., ```python ... ```).
The provided solution should print the correct result for the problem
"""


# Chain of Thought Prompt
PROMPT_COT = """\
You are a mathematical problem solver. Your task is to solve the given problem step by step.

Problem:
{problem}

Begin by outlining your plan for solving the problem before you write any code. First, understand the problem by breaking it down into smaller components. Second, devise a plan for solving each component. Third, implement your solution in Python code. Fourth, make sure that we print the result, and the parameters are actually correct. Finally, verify the solution works by tracing through the code with a simple example.

Present your final solution as a Python code block using triple backticks and the word 'python' (i.e., ```python ... ```) with no additional explanation. Only the last code block will be evaluated, so make sure it's complete and correct.
"""

# Role-playing Prompt
PROMPT_ROLE = """\
You are a world-class mathematician and programmer with expertise in algorithm design. Your team is counting on you to solve an important mathematical problem with efficient Python code.

The problem you need to solve is:
{problem}

As a professional, begin by briefly planning your approach before you start coding. Approach this as a professional:
1. Analyze the mathematical principles involved
2. Design an optimal algorithm considering time and space complexity
3. Implement a clean, efficient solution in Python
4. Make sure to print the result in the solution

Your final solution must be presented as a Python code block using triple backticks and the word 'python' (i.e., ```python ... ```) in markdown format. Only your final code block will be evaluated, so ensure it correctly solves the problem.
"""

# Few-shot Learning Prompt
PROMPT_FEW_SHOT = """\
You are an AI trained to solve mathematical problems with Python. Here are some examples of problems and their solutions:

Before you begin coding, think through and plan your approach to the problem.

Example 1:
Problem: Find the sum of all multiples of 3 or 5 below 1000.
Solution:
```python
def solution():
    total = 0
    for i in range(1000):
        if i % 3 == 0 or i % 5 == 0:
            total += i
    return total

print(solution())
```

Example 2:
Problem: Find the largest palindrome made from the product of two 3-digit numbers.
Solution:
```python
def solution():
    largest = 0
    for i in range(100, 1000):
        for j in range(100, 1000):
            product = i * j
            if product > largest and str(product) == str(product)[::-1]:
                largest = product
    return largest

print(solution())
```

Now solve this problem:
{problem}

Provide only the Python code that solves the problem in a markdown code block using triple backticks and the word 'python' (i.e., ```python ... ```). No explanations needed, only your final solution will be evaluated.
"""

# Structured Reasoning Prompt
PROMPT_STRUCTURED = """\
You are tasked with solving the following mathematical problem:
{problem}

Start by outlining your plan and reasoning before you write any code. Solve this problem by following these exact steps:
1. UNDERSTAND: Identify the key mathematical concepts and constraints in the problem.
2. PLAN: Outline an algorithm that can solve this problem efficiently.
3. CODE: Implement your algorithm in Python.
4. VERIFY: Mentally trace through your code with a simple test case. Verifying that we indeed printed the correct result.

Your final submission should be just the Python code in a markdown code block using triple backticks and the word 'python' (i.e., ```python ... ```). Remember that only this final code block will be evaluated, so ensure it's correct and complete.
"""

# Constraints-focused Prompt
PROMPT_CONSTRAINTS = """\
You will solve a mathematical problem with specific constraints.

Problem to solve:
{problem}

Before you start coding, briefly plan your approach to ensure you meet all constraints.
Your solution MUST:
- Be written in Python
- Be as efficient as possible (consider time and space complexity)
- Include only the necessary code to solve the problem
- Be placed in a markdown code block using triple backticks and the word 'python' (i.e., ```python ... ```)
- Avoid using libraries unless absolutely necessary

You will be evaluated ONLY on your final code block, so make sure it's the complete solution. Do not include explanations - just the final, working Python code. Make sure to include the function call and print the result at the end of your code.
"""

# Expert Prompt
PROMPT_EXPERT = """\
You are an expert in mathematics and programming. Your task is to first devise and then implement solutions for complex mathematical problems using Python.

The problem description:
{problem}

Begin by planning your approach, considering the mathematical concepts and possible algorithms, before you write any code. To solve this problem with optimal Python code. you must:
1. Analyze the underlying mathematical concepts
2. Implement an efficient algorithm

Remember that only the final code block will be evaluated, so you must make sure it is complete and correct. The code should be provided in markdown format using triple backticks and the word 'python' (i.e., ```python ... ```). Make sure to include the function call and print the result at the end of your code.

Think step by step and ensure your solution is correct.
"""

PROMPT_LEAST_TO_MOST = """\
You are tasked with solving a mathematical problem using the Least-to-Most approach.

Problem:
{problem}

Start by planning your approach: identify the simplest subproblem and how you will build up to the full solution before you begin coding.
Follow these steps:
1. Start by identifying the simplest subproblem related to the main problem.
2. Solve this simple subproblem first.
3. Gradually build on top of simpler solutions to tackle increasingly complex parts.
4. Combine all parts carefully into a final complete solution.
5. Implement your final solution in Python.

Only your last code block in markdown format using triple backticks and the word 'python' (i.e., ```python ... ```) will be evaluated, so ensure it is correct, complete, and prints the final result.
"""

PROMPT_SELF_CONSISTENCY = """\
You are a mathematical problem solver using the Self-Consistency method.

Problem:
{problem}

Before you write any code, plan your approach by considering multiple reasoning paths and selecting the most consistent one.
Approach:
1. Think through multiple possible reasoning paths or solution methods.
2. For each path, mentally verify if it leads to a correct solution.
3. Choose the reasoning path that is most logically consistent and reliable.
4. Implement the solution based on the most consistent reasoning.

Provide only the final Python code inside a markdown code block using triple backticks and the word 'python' (i.e., ```python ... ```). Only the final code block will be evaluated, so make sure it prints the correct result and is complete.
"""

PROMPT_COMPREHENSIVE_EXPERT = """\
You are an expert mathematician and programmer. Your task is to solve a complex mathematical problem using a variety of techniques. Approach this problem systematically and efficiently.

Problem:
{problem}

Begin by planning your approach: break down the problem, consider multiple solution paths, and decide on the best strategy before you start coding.
Follow these steps in order:
1. **Understand** the problem by breaking it down into its smallest components (Least-to-Most).
2. **Analyze** multiple reasoning paths and solutions (Self-Consistency), ensuring each approach is logically sound and consistent.
3. **Plan** your algorithm by combining insights from all valid paths and deciding on the most optimal approach.
4. **Implement** the solution incrementally by solving simpler subproblems first and building towards the final solution (Least-to-Most).
5. **Verify** the correctness of your solution by mentally walking through your code with a simple test case. Ensure that the solution works with various test cases to confirm its reliability (Self-Consistency).

Present only your final Python code as a markdown code block using triple backticks and the word 'python' (i.e., ```python ... ```). This code must:
- Solve the problem correctly and efficiently
- Be structured to follow the steps outlined above
- Print the correct result when executed

The final code block will be evaluated only, so ensure it is complete, correct, and properly formatted.
"""

# Minimal but fair baseline prompt
PROMPT_BASELINE = """\
Solve this math problem with Python code:
{problem}

Provide your solution as a Python code block in markdown format using triple backticks and the word 'python' (i.e., ```python ... ```).
Only the final code block will be evaluated.
The provided solution should print the correct result for the problem
"""

# Update the prompts collection to include the baseline
PROMPTS = [
    ("simple", PROMPT_SIMPLE),
    ("cot", PROMPT_COT),
    ("role", PROMPT_ROLE),
    ("few-shot", PROMPT_FEW_SHOT),
    ("structured", PROMPT_STRUCTURED),
    ("constraints", PROMPT_CONSTRAINTS),
    ("least-to-most", PROMPT_LEAST_TO_MOST),
    ("prompt-self-consistency", PROMPT_SELF_CONSISTENCY),
    ("expert", PROMPT_EXPERT),
    ("comprehensive-expert", PROMPT_COMPREHENSIVE_EXPERT),
]
