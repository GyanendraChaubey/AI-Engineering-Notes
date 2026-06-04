# Generative AI Engineer (Part 1) — Interview 9

**Q: Find the name of the student with the 2nd highest marks from a dictionary, without using sort or sorted.**

- The task is to find the student with the second highest marks from a dictionary of student names and their marks.
- We are not allowed to use `sort` or `sorted`.
- The optimal approach is to:
 - Traverse the dictionary to find the highest mark.
 - Traverse again to find the highest mark less than the maximum (i.e., the second highest).
 - Then, find the student(s) with that second highest mark.

**🔑 Key Steps**:
- Initialize variables to track the highest and second highest marks.
- Loop through the dictionary to find the highest mark.
- Loop again to find the second highest mark.
- Loop again to find the student with the second highest mark.

**💻 Code**:
```python
students = {"Sam": 29, "Goutham": 90, "Komal": 56, "Priya": 89, "Imran": 99} # Student marks dictionary

max_marks = float('-inf') # Initialize max_marks to negative infinity
second_max = float('-inf') # Initialize second_max to negative infinity

# First pass: Find the highest marks
for marks in students.values(): # Iterate over all marks
 if marks > max_marks: # If current marks are greater than max_marks
 max_marks = marks # Update max_marks

# Second pass: Find the second highest marks
for marks in students.values(): # Iterate over all marks again
 if marks > second_max and marks < max_marks: # If marks are between second_max and max_marks
 second_max = marks # Update second_max

# Third pass: Find the student with the second highest marks
for name, marks in students.items(): # Iterate over all name, marks pairs
 if marks == second_max: # If marks equal to second_max
 print(name) # Print the student's name (output: Goutham)
 break # Exit after finding the first match
```

**💡 Explanation**:
- **Step 1:** Find the maximum marks by iterating through all values.
- **Step 2:** Find the highest marks less than the maximum (second highest).
- **Step 3:** Find and print the student name with the second highest marks.
- **Time Complexity:** O(n) (three passes, but each is O(n)).
- **Space Complexity:** O(1) (constant extra space).
- This approach avoids sorting and is efficient for small to medium-sized dictionaries.

---

**Q: Write a Python program to print the name of the student with the second highest marks from a dictionary.**

- The task is to find the student with the second highest marks from a dictionary of student names and their marks, without using sorting functions.
- The approach is:
 - First, find the highest mark.
 - Then, find the highest mark less than the maximum (second highest).
 - Finally, print the student name with that mark.

**🔑 Key Steps**:
- Loop through the dictionary to find the maximum marks.
- Loop again to find the second highest marks.
- Loop again to print the student with the second highest marks.

**💻 Code**:
```python
students = {"Sam": 29, "Goutham": 90, "Komal": 56, "Priya": 89, "Imran": 99} # Student marks dictionary

max_marks = float('-inf') # Initialize max_marks to negative infinity
second_max = float('-inf') # Initialize second_max to negative infinity

# First pass: Find the highest marks
for marks in students.values(): # Iterate over all marks
 if marks > max_marks: # If current marks are greater than max_marks
 max_marks = marks # Update max_marks

# Second pass: Find the second highest marks
for marks in students.values(): # Iterate over all marks again
 if marks > second_max and marks < max_marks: # If marks are between second_max and max_marks
 second_max = marks # Update second_max

# Third pass: Find the student with the second highest marks
for name, marks in students.items(): # Iterate over all name, marks pairs
 if marks == second_max: # If marks equal to second_max
 print(name) # Print the student's name (output: Goutham)
 break # Exit after finding the first match
```

**💡 Explanation**:
- The code uses three simple loops to avoid sorting.
- First, it finds the maximum marks.
- Second, it finds the highest marks less than the maximum (second highest).
- Third, it prints the name of the student with the second highest marks.
- Time complexity is O(n), space complexity is O(1).
- This approach is efficient and easy to explain in an interview.

---

**Q: Fix the code to print the name of the student with the second highest marks from a dictionary (without using sort/sorted).**

- The error is due to this line: `for name, marks in students.values():` 
 `students.values()` only returns the marks, not (name, marks) pairs, so unpacking fails.
- You should use `students.items()` to get both name and marks.
- Also, initialize `max_marks` and `second_max` to `float('-inf')` for correctness.

**🔑 Key Steps**:
- Use `students.items()` when you need both name and marks.
- Initialize `max_marks` and `second_max` to negative infinity.
- Loop through the dictionary three times: first for max, second for second max, third to print the name.

**💻 Code**:
```python
students = {"Sam": 29, "Goutham": 90, "Komal": 56, "Priya": 89, "Imran": 99} # Student marks dictionary

max_marks = float('-inf') # Initialize max_marks to negative infinity
second_max = float('-inf') # Initialize second_max to negative infinity

# First loop: Find the highest marks
for marks in students.values(): # Iterate over all marks
 if marks > max_marks: # If current marks are greater than max_marks
 max_marks = marks # Update max_marks

# Second loop: Find the second highest marks
for marks in students.values(): # Iterate over all marks again
 if marks > second_max and marks < max_marks: # If marks are between second_max and max_marks
 second_max = marks # Update second_max

# Third loop: Find the student with the second highest marks
for name, marks in students.items(): # Iterate over all name, marks pairs
 if marks == second_max: # If marks equal to second_max
 print(name) # Print the student's name (output: Goutham)
 break # Exit after finding the first match
```

**💡 Explanation**:
- The main issue was using `.values()` instead of `.items()` when unpacking name and marks.
- Initializing with `float('-inf')` ensures correctness for any integer values.
- This approach is O(n) time and O(1) space, and avoids sorting.
- The output will be `Goutham`, which is the student with the second highest marks.

---

**Q: Modify the code to print all student names with the second highest marks, including duplicates.**

- The current code only prints the first student with the second highest marks.
- If multiple students have the same second highest marks, all their names should be printed.
- To achieve this, after finding the second highest marks, loop through all students and collect names where marks match the second highest.
- Print all such names.

**🔑 Key Steps**:
- Find the maximum marks.
- Find the second highest marks (less than max).
- Collect all student names with marks equal to the second highest.
- Print all such names.

**💻 Code**:
```python
students = {"Sam": 29, "Goutham": 89, "Komal": 56, "Priya": 89, "Imran": 99} # Student marks dictionary

max_marks = float('-inf') # Initialize max_marks to negative infinity
second_max = float('-inf') # Initialize second_max to negative infinity

# First loop: Find the highest marks
for marks in students.values(): # Iterate over all marks
 if marks > max_marks: # If current marks are greater than max_marks
 max_marks = marks # Update max_marks

# Second loop: Find the second highest marks
for marks in students.values(): # Iterate over all marks again
 if marks > second_max and marks < max_marks: # If marks are between second_max and max_marks
 second_max = marks # Update second_max

# Third loop: Collect all student names with the second highest marks
second_highest_students = [] # List to store names with second highest marks
for name, marks in students.items(): # Iterate over all name, marks pairs
 if marks == second_max: # If marks equal to second_max
 second_highest_students.append(name) # Add name to the list

# Print all names with the second highest marks
for name in second_highest_students: # Iterate over the list of names
 print(name) # Print each name (output: Goutham, Priya)
```

**💡 Explanation**:
- The code now collects all names with the second highest marks in a list.
- It prints each name, so if there are duplicates (e.g., both Goutham and Priya have 89), both are printed.
- This approach is still O(n) time and O(n) space (for the list of names).
- It handles all edge cases, including multiple students with the same second highest marks.

---


**Q: Why are you migrating from Elasticsearch to OpenSearch for a RAG project?**

- We are migrating from Elasticsearch to OpenSearch primarily to address scalability, cost, and flexibility requirements for the KGPT (Knowledge GPT) project.
- OpenSearch is open-source and community-driven, which eliminates licensing restrictions and provides more control over deployment and customization.
- As our data volume and user base have grown, OpenSearch offers better scalability for handling large-scale vector search and semantic retrieval workloads.
- OpenSearch integrates natively with AWS services, making it easier to manage, scale, and secure within our existing cloud infrastructure.
- The migration also allows us to leverage advanced features like native k-NN vector search (using HNSW or Faiss engines), improved bulk ingestion APIs, and robust retry logic for high-throughput document indexing.
- Overall, this shift supports our need for a cost-effective, scalable, and enterprise-ready vector database solution as the KGPT platform continues to expand.

---
