from llm_output_parser import extract_code_blocks, extract_last_code_block

# Test case 1: Multiple code blocks with different language tags
test_text1 = """Here's a Python function:

```python
def hello_world():
    print("Hello, World!")
```

And here's a JavaScript function:

```javascript
function helloWorld() {
    console.log("Hello, World!");
}
```

Finally, here's the code you should use:

```
final code block without language tag
with multiple lines
```
"""

# Test case 2: Single code block
test_text2 = """Here's the solution to your problem:

  ```python
def sum_multiples_of_3_or_5(n):
    sum = 0
    for i in range(n):
        if i % 3 == 0 or i % 5 == 0:
            sum += i
    return sum

print(sum_multiples_of_3_or_5(1000))
```

This solution uses a simple yet efficient approach. It iterates through the numbers from 1 to 1000 and checks if each number is a multiple of 3 or 5. If it is, it adds it to the sum. The time complexity of this approach is O(n), where n is the given number. The space complexity is O(1) because only the sum is stored.

The output of the above code is:
```
233168
```
This is the sum of all the multiples of 3 or 5 below 1000.
"""

# Test case 3: No code blocks
test_text3 = "This is just some text without any code blocks."

# Test case 4: Empty code block
test_text4 = """Here's an empty code block:

```

```
"""

# Run the tests
print("Test 1 - Multiple code blocks:")
result1 = extract_last_code_block(test_text1)
print(f"Result: {result1}")
print()

print("Test 2 - Single code block:")
result2 = extract_code_blocks(test_text2)
print(f"Result: {len(result2)}")
print(f"Result: {result2}")
print()

print("Test 3 - No code blocks:")
result3 = extract_last_code_block(test_text3)
print(f"Result: {result3}")
print()

print("Test 4 - Empty code block:")
result4 = extract_last_code_block(test_text4)
print(f"Result: {result4}")

