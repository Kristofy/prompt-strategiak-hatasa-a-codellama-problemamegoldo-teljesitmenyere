"""
Project Euler Data Processing Module

This module handles downloading and processing Project Euler problem data
from multiple sources and merges them into a unified dataset.
"""

from dataclasses import dataclass, field
from typing import Literal, Union, List, Dict, Optional, Any
import logging
import os
import json
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Centralized configuration for the application."""
    # File paths
    output_dir: Path = Path(".")
    output_json_path: Path = field(default_factory=lambda: Path("euler_problems.json"))
    solutions_md_path: Path = field(default_factory=lambda: Path("Solutions.md"))
    
    # Dataset URLs
    kaggle_dataset_id: str = "patrickgendotti/project-euler-full-problem-set"
    solutions_md_url: str = "https://raw.githubusercontent.com/lucky-bai/projecteuler-solutions/refs/heads/master/Solutions.md"
    
    # Processing options
    skip_header_lines: int = 4
    merge_solutions: bool = True
    
    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(exist_ok=True)
        self.output_json_path = self.output_dir / self.output_json_path


@dataclass
class Solution:
    """Represents a solution to a Project Euler problem."""
    problem_id: int
    answer: Union[int, float, str, None]
    answer_type: Literal["Integer", "Float", "String", "Missing"]

    def __post_init__(self):
        """Validate and set the answer type."""
        if self.answer is None:
            self.answer_type = "Missing"
        elif isinstance(self.answer, int):
            self.answer_type = "Integer"
        elif isinstance(self.answer, float):
            self.answer_type = "Float"
        elif isinstance(self.answer, str):
            self.answer_type = "String"
        else:
            raise ValueError(f"Invalid answer type: {type(self.answer)}")


@dataclass
class EulerProblem:
    """Represents a Project Euler problem with metadata and solution."""
    id: int
    title: str
    subtitle: str
    content: str
    html_content: str
    release_date: str
    solved_by_count: int
    difficulty: int
    solution: Optional[Union[int, float, str]] = None
    
    def __post_init__(self):
        """Convert release_date string to datetime object for easier handling."""
        if isinstance(self.release_date, str):
            try:
                self.release_date = datetime.fromisoformat(self.release_date)
            except ValueError:
                logger.warning(f"Invalid date format for problem {self.id}: {self.release_date}")
    
    @property
    def is_solved(self) -> bool:
        """Check if the problem has a solution."""
        return self.solution is not None and (
            isinstance(self.solution, (int, float)) or 
            (isinstance(self.solution, str) and len(self.solution.strip()) > 0)
        )
    
    def check_answer(self, answer_input: str) -> bool:
        """
        Check if the provided answer matches the solution.
        
        Args:
            answer_input: User's answer as a string
            
        Returns:
            bool: True if the answer is correct, False otherwise
        """
        if not self.is_solved:
            logger.warning(f"Cannot check answer for problem {self.id}: No solution available")
            return False
        
        # Trim input and handle empty responses
        answer_input = answer_input.strip() if isinstance(answer_input, str) else ""
        if not answer_input:
            return False
        
        # Get the stored solution
        solution = self.solution
        
        # Check based on solution type
        if isinstance(solution, int):
            # For integers, attempt exact conversion and comparison
            try:
                user_answer_int = int(answer_input)
                return user_answer_int == solution
            except (ValueError, TypeError):
                return False
                
        elif isinstance(solution, float):
            # For floats, use relative tolerance comparison
            try:
                user_answer_float = float(answer_input)
                # Using math.isclose with relative tolerance of 1e-9
                import math
                return math.isclose(user_answer_float, solution, rel_tol=1e-9)
            except (ValueError, TypeError):
                return False
                
        elif isinstance(solution, str):
            # For strings, compare lowercase versions
            return answer_input.lower() == solution.lower()
            
        # Fallback case - unknown solution type
        return False


class DataDownloader:
    """Handles downloading data from external sources."""
    
    def __init__(self, config: AppConfig):
        """Initialize with application configuration."""
        self.config = config
    
    def download_solutions_md(self) -> bool:
        """
        Downloads the Solutions.md file from GitHub.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.get(self.config.solutions_md_url, timeout=30)
            response.raise_for_status()
            
            with open(self.config.solutions_md_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded Solutions.md from {self.config.solutions_md_url}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download Solutions.md: {str(e)}")
            return False
    
    def download_kaggle_dataset(self) -> Optional[str]:
        """
        Downloads the Project Euler dataset from Kaggle.
        
        Returns:
            Optional[str]: Path to downloaded dataset or None if failed
        """
        try:
            import kagglehub
            path = kagglehub.dataset_download(self.config.kaggle_dataset_id)
            logger.info(f"Downloaded Kaggle dataset to: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {str(e)}")
            logger.info("Make sure you've authenticated with Kaggle using kaggle.json")
            return None


class SolutionsParser:
    """Parses solutions from the Solutions.md file."""
    
    def __init__(self, config: AppConfig):
        """Initialize with application configuration."""
        self.config = config
    
    def parse(self) -> Dict[int, Solution]:
        """
        Parse the Solutions.md file and extract problem IDs and answers.
        
        Returns:
            Dict[int, Solution]: Dictionary mapping problem IDs to Solution objects
        """
        solutions = {}
        
        try:
            if not os.path.exists(self.config.solutions_md_path):
                logger.error(f"File not found: {self.config.solutions_md_path}")
                return solutions
                
            with open(self.config.solutions_md_path, "r", encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header lines
            solution_lines = lines[self.config.skip_header_lines:] if len(lines) > self.config.skip_header_lines else []
            
            processed_count = 0
            skipped_count = 0
            
            for line_number, line in enumerate(solution_lines, start=self.config.skip_header_lines + 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    # Parse line to extract problem ID and answer
                    # Format example: "677. 984183023"
                    if "." not in line:
                        skipped_count += 1
                        logger.debug(f"Line {line_number} skipped: No delimiter found in '{line}'")
                        continue
                        
                    parts = line.split(".", 1)
                    if len(parts) != 2:
                        skipped_count += 1
                        logger.debug(f"Line {line_number} skipped: Invalid format '{line}'")
                        continue
                    
                    # Extract and validate problem ID
                    problem_id_str = parts[0].strip()
                    try:
                        problem_id = int(problem_id_str)
                    except ValueError:
                        skipped_count += 1
                        logger.debug(f"Line {line_number} skipped: Invalid problem ID '{problem_id_str}'")
                        continue
                    
                    # Extract and parse answer
                    answer_str = parts[1].strip()
                    answer = None
                    
                    if not answer_str:
                        # Empty answer
                        answer = None
                    elif "." in answer_str and not answer_str.startswith(("e", "E")):
                        # Try parsing as float
                        try:
                            answer = float(answer_str)
                        except ValueError:
                            # If it can't be parsed as float, keep as string
                            answer = answer_str
                    else:
                        # Try parsing as integer
                        try:
                            answer = int(answer_str)
                        except ValueError:
                            # If it can't be parsed as integer, keep as string
                            answer = answer_str
                    
                    # Create Solution object
                    solutions[problem_id] = Solution(
                        problem_id=problem_id,
                        answer=answer,
                        answer_type="Missing" if answer is None else "Integer" if isinstance(answer, int) else "Float" if isinstance(answer, float) else "String"
                    )
                    processed_count += 1
                    
                except Exception as e:
                    skipped_count += 1
                    logger.error(f"Error processing line {line_number}: '{line}' - {str(e)}")
            
            logger.info(f"Processed {processed_count} solutions, skipped {skipped_count} lines")
        
        except Exception as e:
            logger.error(f"Error reading solutions file: {str(e)}")
        
        return solutions


class KaggleDataProcessor:
    """Processes Project Euler problems data from Kaggle dataset."""
    
    def __init__(self, config: AppConfig):
        """Initialize with application configuration."""
        self.config = config
    
    def process_csv(self, dataset_path: str) -> List[EulerProblem]:
        """
        Process CSV data from Kaggle dataset.
        
        Args:
            dataset_path (str): Path to the downloaded dataset
            
        Returns:
            List[EulerProblem]: List of processed EulerProblem objects
        """
        try:
            # Find the CSV file in the dataset directory
            csv_file = None
            for file_name in os.listdir(dataset_path):
                if file_name.endswith('.csv'):
                    csv_file = os.path.join(dataset_path, file_name)
                    break
            
            if not csv_file:
                logger.error("No CSV file found in the dataset directory")
                return []
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Convert the dataframe to a list of dictionaries
            problems_raw = df.to_dict('records')
            
            # Convert the list of dictionaries to a list of EulerProblem instances
            problems = [EulerProblem(**problem) for problem in problems_raw]
            
            logger.info(f"Processed {len(problems)} problems from Kaggle dataset")
            return problems
            
        except Exception as e:
            logger.error(f"Error processing Kaggle dataset: {str(e)}")
            return []


class DataProcessor:
    """Main class to orchestrate downloading, processing and merging datasets."""
    
    def __init__(self, config: AppConfig = None):
        """Initialize with optional custom configuration."""
        self.config = config if config is not None else AppConfig()
        self.downloader = DataDownloader(self.config)
        self.solutions_parser = SolutionsParser(self.config)
        self.kaggle_processor = KaggleDataProcessor(self.config)
        
    def process(self) -> List[EulerProblem]:
        """
        Main processing function to download, parse and merge data.
        
        Returns:
            List[EulerProblem]: List of processed EulerProblem objects with solutions
        """
        # Download Solutions.md
        if not os.path.exists(self.config.solutions_md_path):
            self.downloader.download_solutions_md()
            
        # Download Kaggle dataset
        dataset_path = self.downloader.download_kaggle_dataset()
        if not dataset_path:
            logger.error("Failed to download Kaggle dataset, cannot proceed")
            return []
            
        # Process Kaggle dataset
        problems = self.kaggle_processor.process_csv(dataset_path)
        if not problems:
            logger.error("Failed to process problems from Kaggle dataset")
            return []
            
        # Parse solutions if merge option is enabled
        if self.config.merge_solutions:
            solutions = self.solutions_parser.parse()
            if solutions:
                self._merge_solutions(problems, solutions)
            
        # Save processed data to JSON
        self._save_to_json(problems)
        
        return problems
    
    def _merge_solutions(self, problems: List[EulerProblem], solutions: Dict[int, Solution]) -> int:
        """
        Merge solutions into problems.
        
        Args:
            problems: List of EulerProblem objects
            solutions: Dictionary of Solution objects
            
        Returns:
            int: Number of problems updated
        """
        updated_count = 0
        
        try:
            for problem in problems:
                if problem.id in solutions:
                    problem.solution = solutions[problem.id].answer
                    updated_count += 1
                    
            logger.info(f"Merged solutions into {updated_count} problems")
            
        except Exception as e:
            logger.error(f"Error merging solutions: {str(e)}")
        
        return updated_count
    
    def _save_to_json(self, problems: List[EulerProblem]) -> bool:
        """
        Save problems to JSON file.
        
        Args:
            problems: List of EulerProblem objects
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert problems to dictionaries
            problems_dict = []
            for problem in problems:
                problem_dict = {
                    "id": problem.id,
                    "title": problem.title,
                    "subtitle": problem.subtitle,
                    "content": problem.content,
                    "html_content": problem.html_content,
                    "release_date": problem.release_date.isoformat() if hasattr(problem.release_date, 'isoformat') else problem.release_date,
                    "solved_by_count": problem.solved_by_count,
                    "difficulty": problem.difficulty,
                    "solution": problem.solution
                }
                problems_dict.append(problem_dict)
            
            # Write the JSON structure to a file
            with open(self.config.output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(problems_dict, json_file, indent=4)
            
            logger.info(f"JSON file created at: {self.config.output_json_path}")
            logger.info(f"Contains {len(problems)} Project Euler problems.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
            return False
    
    def get_statistics(self, problems: List[EulerProblem]) -> Dict[str, Any]:
        """
        Generate statistics about the processed problems.
        
        Args:
            problems: List of EulerProblem objects
            
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        stats = {
            "total_problems": len(problems),
            "solved_problems": sum(1 for p in problems if p.is_solved),
            "unsolved_problems": sum(1 for p in problems if not p.is_solved),
        }
        
        if stats["solved_problems"] > 0:
            stats["solution_types"] = {
                "integer": sum(1 for p in problems if p.is_solved and isinstance(p.solution, int)),
                "float": sum(1 for p in problems if p.is_solved and isinstance(p.solution, float)),
                "string": sum(1 for p in problems if p.is_solved and isinstance(p.solution, str)),
            }
            
            avg_difficulty = sum(p.difficulty for p in problems if p.is_solved) / stats["solved_problems"]
            stats["avg_difficulty_solved"] = round(avg_difficulty, 2)
        
        return stats


def main():
    """Main entry point for the application."""
    try:
        # Create processor with default config
        processor = DataProcessor()
        
        # Process data
        problems = processor.process()
        
        # Generate and display statistics
        if problems:
            stats = processor.get_statistics(problems)
            logger.info("Processing complete. Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for subkey, subvalue in value.items():
                        logger.info(f"    {subkey}: {subvalue}")
                else:
                    logger.info(f"  {key}: {value}")
            
            # Display some example problems
            if len(problems) > 0:
                logger.info("\nExample problems:")
                for problem in problems[:3]:
                    solution_info = f"Solution: {problem.solution}" if problem.is_solved else "No solution"
                    logger.info(f"  Problem #{problem.id}: {problem.title} - {solution_info}")
        else:
            logger.warning("No problems were processed.")
            
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
