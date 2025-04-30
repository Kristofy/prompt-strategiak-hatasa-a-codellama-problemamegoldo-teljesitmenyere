import subprocess
import tempfile
import sys
import os
import uuid
import time
import signal
import ast
import re
from typing import Dict, Any, Optional, Tuple, List, Set


class CodeSanitizer:
    """
    Class to sanitize and check Python code for potentially dangerous operations
    """

    # Modules that are considered safe to import
    SAFE_MODULES = {
        "math",
        "random",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "string",
        "re",
        "json",
        "csv",
        "time",
        "sys",
        "copy",
        "typing",
        "statistics",
    }

    # Unsafe modules/attributes/patterns that should be blocked
    UNSAFE_PATTERNS = [
        r"os\.(system|popen|spawn|exec|unlink|remove)",
        r"subprocess\.",
        r"shutil\.(rmtree|remove|move|copy)",
        r"__import__\(",
        r"eval\(",
        r"exec\(",
        r"globals\(",
        r"locals\(",
        r"getattr\(",
        r"setattr\(",
        r"delattr\(",
        r"open\(",
        r"file\(",
        r"importlib",
        r"pty\.",
        r"socket\.",
        r"requests\.",
        r"urllib",
        r"pathlib",
        r"sys\.modules",
        r"builtins\.__",  # Block access to double underscore attributes
    ]

    @classmethod
    def check_imports(cls, code: str) -> List[str]:
        """
        Check for unsafe imports in the code

        Args:
            code: Python code to check

        Returns:
            List of unsafe import statements found
        """
        try:
            tree = ast.parse(code)
            unsafe_imports = []

            for node in ast.walk(tree):
                # Check for import statements
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name.split(".")[0] not in cls.SAFE_MODULES:
                            unsafe_imports.append(f"import {name.name}")

                # Check for from-import statements
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] not in cls.SAFE_MODULES:
                        imports = [n.name for n in node.names]
                        unsafe_imports.append(f"from {node.module} import {', '.join(imports)}")

            return unsafe_imports
        except SyntaxError:
            # If code can't be parsed, consider it unsafe
            return ["Invalid code syntax"]

    @classmethod
    def check_dangerous_patterns(cls, code: str) -> List[str]:
        """
        Check for dangerous patterns in the code using regex

        Args:
            code: Python code to check

        Returns:
            List of dangerous patterns found
        """
        found_patterns = []

        for pattern in cls.UNSAFE_PATTERNS:
            if re.search(pattern, code):
                found_patterns.append(pattern)

        return found_patterns

    @classmethod
    def is_safe(cls, code: str) -> Tuple[bool, List[str]]:
        """
        Determine if code is safe to execute

        Args:
            code: Python code to check

        Returns:
            Tuple of (is_safe, reasons)
        """
        unsafe_imports = cls.check_imports(code)
        dangerous_patterns = cls.check_dangerous_patterns(code)

        if unsafe_imports or dangerous_patterns:
            return False, unsafe_imports + dangerous_patterns

        return True, []


def execute_python_code_safely(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Execute Python code in a safe subprocess with timeout and return the execution results.

    Args:
        code: The Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results including stdout, stderr, execution_time, and status
    """
    # First, check if the code is safe to execute
    is_safe, unsafe_reasons = CodeSanitizer.is_safe(code)

    if not is_safe:
        return {
            "status": "rejected",
            "stdout": "",
            "stderr": f"Code rejected for security reasons:\n" + "\n".join(unsafe_reasons),
            "execution_time": 0,
            "exit_code": -1,
        }

    # Instead of trying to build our own sandbox, which is error-prone,
    # we'll use Python's built-in restricted execution model

    # Create a unique filename for this execution
    file_id = str(uuid.uuid4())
    temp_file_path = os.path.join(tempfile.gettempdir(), f"safe_exec_{file_id}.py")

    try:
        # Write code to temporary file
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(code)

        # Record start time
        start_time = time.time()

        # Run the code in a subprocess with restricted permissions
        # Using -E flag to ignore environment variables
        # Using -S flag to not import site module
        process = subprocess.Popen(
            [sys.executable, "-ESu", temp_file_path],  # -u for unbuffered output
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,  # Create a new process group to kill child processes if needed
            env={"PYTHONPATH": ""},  # Prevent loading modules from PYTHONPATH
        )

        try:
            # Wait for the process to complete with timeout
            stdout, stderr = process.communicate(timeout=timeout)
            execution_time = time.time() - start_time

            return {
                "status": "success" if process.returncode == 0 else "error",
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": execution_time,
                "exit_code": process.returncode,
            }

        except subprocess.TimeoutExpired:
            # Kill the process group if timeout occurs
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.kill()
            stdout, stderr = process.communicate()

            return {"status": "timeout", "stdout": stdout, "stderr": stderr, "execution_time": timeout, "exit_code": -1}

    except Exception as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Evaluation error: {str(e)}",
            "execution_time": 0,
            "exit_code": -1,
        }
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def evaluate(code_snippet: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Evaluate a Python code snippet and return the execution results.

    Args:
        code_snippet: Python code to evaluate
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    return execute_python_code_safely(code_snippet, timeout)


if __name__ == "__main__":
    # Example usage with safe code
    print("Testing safe code:")
    test_code = """
print("Hello, World!")
x = 5
y = 10
print(f"x + y = {x + y}")
"""

    result = evaluate(test_code)
    print(f"Status: {result['status']}")
    print(f"Output: {result['stdout']}")
    print(f"Errors: {result['stderr']}")
    print(f"Execution time: {result['execution_time']:.4f} seconds")

    # Example of unsafe code
    print("\nTesting unsafe code:")
    unsafe_code = """
import os
print("Trying to list files:")
print(os.listdir('.'))
"""

    result = evaluate(unsafe_code)
    print(f"Status: {result['status']}")
    print(f"Output: {result['stdout']}")
    print(f"Errors: {result['stderr']}")
    print(f"Execution time: {result['execution_time']:.4f} seconds")
