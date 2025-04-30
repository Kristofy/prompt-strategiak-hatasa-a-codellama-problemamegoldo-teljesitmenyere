from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import pickle
import sqlite3
import os
import prompts
from openai import OpenAI
import json
import math  # Import math for isclose
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, Dict, Any, Literal, Union
from llm_output_parser import extract_last_code_block, extract_code_blocks
from evaluator import evaluate
from prettytable import PrettyTable
import csv


@dataclass
class Problem:
    """Represents a Project Euler problem."""

    id: int
    title: str
    subtitle: str
    content: str
    html_content: str
    release_date: str
    solved_by_count: int
    difficulty: int
    solution: Union[int, float, str]

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Problem):
            return NotImplemented
        return self.id == other.id

    def as_prompt(self) -> str:
        """
        Converts the problem instance to a prompt string.
        """
        return f"Problem {self.title} -- {self.subtitle}\n{self.content}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Problem":
        """
        Creates a Problem instance from a dictionary.
        Returns None if the solution is missing.
        """

        # You might want to add more robust validation/error handling here
        return cls(
            id=data["id"],
            title=data["title"],
            subtitle=data["subtitle"],
            content=data["content"],
            html_content=data["html_content"],
            release_date=data["release_date"],
            solved_by_count=data["solved_by_count"],
            difficulty=data["difficulty"],
            solution=data["solution"],
        )

    @classmethod
    def from_json(cls, file_path: str) -> list["Problem"]:
        """
        Creates a Problem instance from a JSON string.
        Returns None if the solution is missing in the JSON data.
        """
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            print(f"Loaded {len(data)} problems from {file_path}.")
            problems = [Problem.from_dict(item) for item in data]
            return [problem for problem in problems if problem.solution is not None]  # Filter out None values
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}")

    def verify_solution(self, proposed_solution: Any, tolerance: float = 1e-9) -> bool:
        """
        Verifies if the proposed solution matches the actual solution.

        Args:
            proposed_solution: The solution attempt to verify.
            tolerance: The absolute tolerance for float comparisons.

        Returns:
            True if the proposed solution is correct, False otherwise.
        """
        actual_solution = self.solution

        # Check type compatibility first if necessary, or rely on comparison operators
        # For simplicity, we assume proposed_solution can be compared

        if isinstance(actual_solution, float):
            # Check if proposed solution can be converted to float
            try:
                proposed_float = float(proposed_solution)
                return math.isclose(actual_solution, proposed_float, abs_tol=tolerance)
            except (ValueError, TypeError):
                return False  # Cannot compare if proposed isn't float-like
        elif isinstance(actual_solution, int):
            # Check if proposed solution can be converted to int
            try:
                # Allow comparison if proposed is float but numerically equal to int
                if isinstance(proposed_solution, float) and proposed_solution.is_integer():
                    return actual_solution == int(proposed_solution)
                # Direct integer comparison
                return actual_solution == int(proposed_solution)
            except (ValueError, TypeError):
                return False  # Cannot compare if proposed isn't int-like
        elif isinstance(actual_solution, str):
            # Direct string comparison
            return str(actual_solution) == str(proposed_solution)
        else:
            # Fallback for any other types (though current definition limits it)
            # Or raise an error for unsupported types
            print(f"Warning: Unsupported solution type for verification: {type(actual_solution)}")
            return actual_solution == proposed_solution


def cached(
    permanent: Union[bool, Callable] = True,
    pred: Callable | None = None,
    backend: Literal["file", "sqlite"] = "file",
):
    CACHE_DIR: str = str(os.path.join(os.path.dirname(__file__), ".cache"))
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    DB_PATH = os.path.join(CACHE_DIR, "cache.db")

    def cache(__func, __pred, *args, **kwargs):
        SEP = "$|$"
        cache_token = (
            f"{__func.__name__}{SEP}"
            f"{SEP.join(str(arg) for arg in args)}{SEP}"
            f"{SEP.join(str(key) + SEP * 2 + str(val) for key, val in kwargs.items())}"
        )
        hex_hash = hashlib.sha256(cache_token.encode()).hexdigest()
        cache_filename: str = os.path.join(CACHE_DIR, f"{__func.__name__}-{hex_hash}")

        if backend == "sqlite":
            # Ensure table exists
            conn = sqlite3.connect(DB_PATH)
            try:
                conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)")
                cur = conn.execute("SELECT value FROM cache WHERE key=?", (hex_hash,))
                row = cur.fetchone()
                if row:
                    return pickle.loads(row[0])

            finally:
                conn.close()
        else:  # file backend
            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as cache_file:
                    return pickle.load(cache_file)

        result = __func(*args, **kwargs)
        if __pred(result):
            if backend == "sqlite":
                conn = sqlite3.connect(DB_PATH)
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                        (hex_hash, pickle.dumps(result)),
                    )
                    conn.commit()
                finally:
                    conn.close()
            else:
                with open(cache_filename, "wb") as cache_file:
                    pickle.dump(result, cache_file)

        return result

    def clear_cache_for_func(func):
        if backend == "sqlite":
            conn = sqlite3.connect(DB_PATH)
            try:
                conn.execute(
                    "DELETE FROM cache WHERE key LIKE ?",
                    (f"{hashlib.sha256(func.__name__.encode()).hexdigest()}%",),
                )
                conn.commit()
            finally:
                conn.close()
        else:
            for file in os.listdir(CACHE_DIR):
                if file.startswith(f"{func.__name__}-"):
                    os.remove(os.path.join(CACHE_DIR, file))

    if callable(permanent):

        def _wrapper(*args, **kwargs):
            return cache(permanent, lambda _: True, *args, **kwargs)

        return _wrapper
    elif isinstance(permanent, bool) and callable(pred):

        def outer(func):
            if not permanent:
                clear_cache_for_func(func)

            def wrapper(*args, **kwargs):
                return cache(func, pred, *args, **kwargs)

            return wrapper

        return outer
    elif isinstance(permanent, bool) and pred is None:

        def wrapper(func):
            if not permanent:
                clear_cache_for_func(func)

            def inner(*args, **kwargs):
                return cache(func, lambda _: True, *args, **kwargs)

            return inner

        return wrapper
    else:
        raise ValueError("Invalid predicate")


def get_ai():

    # Initialize the client, pointing to the local server
    client = OpenAI(
        base_url="http://localhost:8081/v1",
        api_key="idk",  # Use a placeholder API key if required by your local server
    )

    # # Get the first available model ID
    # try:
    #     models = client.models.list()
    #     if models.data:
    #         model_id = models.data[0].id
    #     else:
    #         print("Error: No models found at the specified endpoint.")
    #         exit()
    # except Exception as e:
    #     print(f"Error connecting to the API or listing models: {e}")
    #     exit()
    #
    def promptable(model: str, system_prompt, prompt, temperature, max_tokens, top_p):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            # Extract the response content
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {e}")
            raise e

    return promptable


# Print statistics about the problems
def print_problem_statistics(problems):
    """Print statistics about the loaded Project Euler problems."""
    if not problems:
        print("No problems loaded.")
        return

    print("\n=== Project Euler Problems Statistics ===")
    print(f"Total problems loaded: {len(problems)}")

    # Difficulty statistics
    difficulties = [p.difficulty for p in problems]
    avg_difficulty = sum(difficulties) / len(difficulties)
    print(f"Average difficulty: {avg_difficulty:.2f}")
    print(f"Min difficulty: {min(difficulties)}")
    print(f"Max difficulty: {max(difficulties)}")

    # Solved by count statistics
    solved_counts = [p.solved_by_count for p in problems]
    avg_solved = sum(solved_counts) / len(solved_counts)
    print(f"Average number of solvers: {avg_solved:.2f}")

    # Find most/least solved problems
    most_solved = max(problems, key=lambda p: p.solved_by_count)
    least_solved = min(problems, key=lambda p: p.solved_by_count)
    print(f"Most solved problem: #{most_solved.id} ({most_solved.title}) - {most_solved.solved_by_count} solvers")
    print(f"Least solved problem: #{least_solved.id} ({least_solved.title}) - {least_solved.solved_by_count} solvers")

    # Group by difficulty ranges
    difficulty_ranges = {"Easy (1-25)": 0, "Medium (26-50)": 0, "Hard (51-75)": 0, "Very Hard (76-100)": 0}

    for p in problems:
        if p.difficulty <= 25:
            difficulty_ranges["Easy (1-25)"] += 1
        elif p.difficulty <= 50:
            difficulty_ranges["Medium (26-50)"] += 1
        elif p.difficulty <= 75:
            difficulty_ranges["Hard (51-75)"] += 1
        else:
            difficulty_ranges["Very Hard (76-100)"] += 1

    print("\nDifficulty Distribution:")
    for range_name, count in difficulty_ranges.items():
        percentage = (count / len(problems)) * 100
        print(f"  {range_name}: {count} problems ({percentage:.1f}%)")


call = get_ai()


@cached(permanent=True, backend="sqlite")
def solve_problem(
    problem: Problem,
    model: str,
    system_prompt: str,
    prompt_template: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
):
    """
    Solves a Project Euler problem using the AI.
    Returns the solution if successful, None otherwise.
    """
    verbose = False

    prompt = prompt_template.format(problem=problem.as_prompt())

    if verbose:
        print(f"\n\n\n\n\n\n\n\n\n\nPrompting AI with:\n{prompt}")

    response = call(
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    if verbose:
        print(f"\n\nAI response: {response}")

    if not response:
        return False

    cbs = extract_code_blocks(response)

    if not cbs:
        if verbose:
            print("\n\nNo code block found in the response.")
        return False

    if verbose:
        print(f"\n\nExtracted code blocks:\n{cbs}")

    for cb in cbs:
        result = evaluate(cb)

        if result.get("status") != "success":
            if verbose:
                print(f"\n\nError in execution: {result.get('stderr')}")
            continue

        if result.get("exit_code") != 0:
            if verbose:
                print(f"\n\nNon-zero exit code: {result.get('exit_code')}")
            continue

        answer = result.get("stdout", "").strip()

        if verbose:
            print(f"\n\nAI's answer: {answer}")
            print(f"Expected answer: {problem.solution}")

        ok = problem.verify_solution(answer, 0.0001)
        if ok:
            return True

    return False


def worker(
    problem: Problem,
    model: str,
    system_prompt: str,
    prompt_template: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> tuple[int, bool]:
    """
    Initialize AI client per process and solve one problem.
    Returns a tuple of (problem id, success flag).
    """
    success = solve_problem(problem, model, system_prompt, prompt_template, temperature, max_tokens, top_p)  # type: ignore
    return problem.id, success


def solve_all_with_single_template(
    problems: list[Problem],
    model: str,
    system_prompt: str,
    prompt_template: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    verbose: bool = False,
    idx: int = 0,
) -> dict[int, bool]:
    results: dict[int, bool] = {}
    with ProcessPoolExecutor(7) as executor:
        futures = {
            executor.submit(worker, p, model, system_prompt, prompt_template, temperature, max_tokens, top_p): p
            for p in problems
        }
        for future in tqdm(
            as_completed(futures),
            total=len(problems),
            desc=f"Solving problems with Template {idx}",
            position=idx,
            leave=True,
        ):
            pid, ok = future.result()
            results[pid] = ok
            if verbose:
                status = "✔" if ok else "✘"
                print(f"Problem {pid}: {status}")
    return results


def solve_all_with_all_templates(
    problems: list[Problem],
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> dict[str, dict[int, bool]]:
    results: dict[str, dict[int, bool]] = {}

    with ProcessPoolExecutor(len(prompts.PROMPTS)) as executor:
        futures = {
            executor.submit(
                solve_all_with_single_template,
                problems,
                model,
                system_prompt,
                template,
                temperature,
                max_tokens,
                top_p,
                idx=idx + 1,
            ): k
            for idx, (k, template) in enumerate(prompts.PROMPTS)
        }

        for future in tqdm(
            as_completed(futures), total=len(prompts.PROMPTS), desc="Prompt Templates", position=0, leave=True
        ):
            k = futures[future]
            results[k] = future.result()

    return results


if __name__ == "__main__":
    # Load and prepare problems
    problems = Problem.from_json("euler_problems.json")
    print_problem_statistics(problems)
    problems.sort(key=lambda p: p.solved_by_count, reverse=True)

    # Filter out only the easy problems
    problems = [p for p in problems if p.difficulty <= 25]
    problems = problems[:100]

    temperature = 0.95
    top_p = 0.95

    results = solve_all_with_all_templates(
        problems,
        "meta-llama/CodeLlama-34b-Instruct-hf",
        "You are a helpful assistant that solves a provided problem using Python.",
        temperature=temperature,
        max_tokens=16384,
        top_p=top_p,
    )

    # Provide a summary of the results
    print("\n=== Summary of Results ===")

    # Difficulty bins
    difficulty_bins = [
        ("Easy (1-25)", 1, 25),
        ("Medium (26-50)", 26, 50),
        ("Hard (51-75)", 51, 75),
        ("Very Hard (76-100)", 76, 100),
    ]

    # Build table
    table = PrettyTable()
    table.field_names = [
        "Prompt Template",
        "Temperature",
        "Top_p",
        "Correct/Total",
        "Easy",
        "Medium",
        "Hard",
        "Very Hard",
    ]

    # Prepare CSV file (reset at start)
    csv_path = "results_summary.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(table.field_names)

    # Only one config in this script, but keep structure for future
    total = len(problems)

    for prompt_name, _ in prompts.PROMPTS:
        if prompt_name not in results:
            continue
        res = results[prompt_name]
        # Map problem id to Problem object
        pid2prob = {p.id: p for p in problems}
        # Count correct
        correct = sum(1 for v in res.values() if v)
        # Difficulty breakdown
        diff_counts = {k: [0, 0] for k, _, _ in difficulty_bins}  # [correct, total]
        for pid, ok in res.items():
            prob = pid2prob.get(pid)
            if not prob:
                continue
            for k, lo, hi in difficulty_bins:
                if lo <= prob.difficulty <= hi:
                    diff_counts[k][1] += 1
                    if ok:
                        diff_counts[k][0] += 1
        row = [
            prompt_name,
            temperature,
            top_p,
            f"{correct}/{total}",
        ]
        for k, _, _ in difficulty_bins:
            c, t = diff_counts[k]
            row.append(f"{c}/{t}" if t else "-")
        table.add_row(row)
        # Append to CSV after each row
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    print(table)
