# Vision Pipeline Engineer — Interview 1

**Q: Find the pair of bounding boxes with the largest overlapping area**

This is a **computational geometry problem** focused on finding maximum overlap between axis-aligned rectangles.

**🎯 Approach**:
• Calculate **intersection area** for every pair of bounding boxes using coordinate geometry
• Track the pair with **maximum overlap area**
• Return indices and overlap value as a tuple

**🔑 Key Steps**:
1. **Iterate through all pairs** of bounding boxes (nested loop with i < j to avoid duplicates)
2. **Calculate intersection rectangle** coordinates:
 - `x_overlap_start = max(x_min1, x_min2)`
 - `y_overlap_start = max(y_min1, y_min2)`
 - `x_overlap_end = min(x_max1, x_max2)`
 - `y_overlap_end = min(y_max1, y_max2)`
3. **Check if valid overlap exists**: `x_overlap_start < x_overlap_end AND y_overlap_start < y_overlap_end`
4. **Compute overlap area**: `(x_overlap_end - x_overlap_start) * (y_overlap_end - y_overlap_start)`
5. **Track maximum** and return result

**💻 Code**:
```python
def find_largest_overlap(bboxes):
 # Initialize variables to track the maximum overlap
 max_overlap = 0
 # Store the indices of boxes with maximum overlap
 result_indices = (None, None)
 
 # Iterate through all pairs of bounding boxes
 for i in range(len(bboxes)):
 # Get first bounding box coordinates
 x_min1, y_min1, x_max1, y_max1 = bboxes[i]
 
 # Compare with all subsequent boxes to avoid duplicate pairs
 for j in range(i + 1, len(bboxes)):
 # Get second bounding box coordinates
 x_min2, y_min2, x_max2, y_max2 = bboxes[j]
 
 # Calculate the intersection rectangle's left boundary
 x_overlap_start = max(x_min1, x_min2)
 # Calculate the intersection rectangle's top boundary
 y_overlap_start = max(y_min1, y_min2)
 # Calculate the intersection rectangle's right boundary
 x_overlap_end = min(x_max1, x_max2)
 # Calculate the intersection rectangle's bottom boundary
 y_overlap_end = min(y_max1, y_max2)
 
 # Check if there is a valid overlap (intersection exists)
 if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
 # Calculate the width of the overlapping region
 overlap_width = x_overlap_end - x_overlap_start
 # Calculate the height of the overlapping region
 overlap_height = y_overlap_end - y_overlap_start
 # Calculate the total overlapping area
 overlap_area = overlap_width * overlap_height
 
 # Update maximum overlap if current overlap is larger
 if overlap_area > max_overlap:
 # Store the new maximum overlap area
 max_overlap = overlap_area
 # Store the indices of the boxes with maximum overlap
 result_indices = (i, j)
 
 # Return None if no overlap exists
 if result_indices[0] is None:
 return (None, None, 0)
 
 # Return the indices and overlap value as a tuple
 return (result_indices[0], result_indices[1], max_overlap)


# Test with the provided example
bboxes = [
 (10, 10, 50, 50),
 (30, 30, 70, 70),
 (100, 100, 150, 150),
]

# Call the function and print result
result = find_largest_overlap(bboxes)
# Display the output
print(f"Result: {result}")
```

**💡 Explanation**:

**Algorithm Logic**:
• Uses **brute force O(n²)** approach to check all pairs - optimal for this problem since we need to examine every combination
• **Intersection calculation** uses standard rectangle overlap formula: overlap exists only when `max(left_edges) < min(right_edges)` AND `max(top_edges) < min(bottom_edges)`
• **Area computation** is straightforward: `width × height` of the intersection rectangle

**Time Complexity**: **O(n²)** where n is the number of bounding boxes
• We check all pairs: n(n-1)/2 comparisons
• Each comparison involves constant-time coordinate calculations

**Space Complexity**: **O(1)**
• Only storing a few variables (max_overlap, result_indices, loop variables)
• No additional data structures proportional to input size

**Edge Cases Handled**:
• **No overlap**: Returns `(None, None, 0)` as specified
• **Multiple boxes**: Correctly identifies the pair with maximum overlap
• **Adjacent boxes** (touching but not overlapping): Correctly returns 0 overlap since condition is `<` not `<=`

**Example Walkthrough** (given input):
• Box 0: (10,10,50,50) vs Box 1: (30,30,70,70) → Overlap: (30,30,50,50) → Area = 20×20 = **400**
• Box 0: (10,10,50,50) vs Box 2: (100,100,150,150) → No overlap
• Box 1: (30,30,70,70) vs Box 2: (100,100,150,150) → No overlap
• **Result**: `(0, 1, 400)` ✓

---

**Q: Analyze the coding problem shown on screen - calculate total area covered by overlapping bounding boxes**

This is a **computational geometry problem** about calculating the **total area covered by a set of overlapping rectangles** (bounding boxes). The key challenge is handling overlaps correctly - you can't just sum all areas because overlapping regions would be counted multiple times.

**🔑 Key Approach - Coordinate Compression + Sweep Line**:

• Use **coordinate compression** to discretize the 2D plane into a grid of non-overlapping cells
• For each cell, check if it's covered by any bounding box
• Sum up areas of all covered cells
• Time complexity: O(n² × m²) where n is number of unique x-coordinates and m is unique y-coordinates
• For small inputs (4 boxes), this is efficient enough

**Alternative approach**: Inclusion-Exclusion Principle (more complex but O(2^n))

---

**💻 Code**:

```python
def calculate_total_area(bboxes):
 # Handle empty input
 if not bboxes:
 return 0
 
 # Extract all unique x and y coordinates
 x_coords = set()
 y_coords = set()
 
 for x1, y1, x2, y2 in bboxes:
 # Add all x coordinates (left and right edges)
 x_coords.add(x1)
 x_coords.add(x2)
 # Add all y coordinates (top and bottom edges)
 y_coords.add(y1)
 y_coords.add(y2)
 
 # Sort coordinates to create grid lines
 x_coords = sorted(x_coords)
 y_coords = sorted(y_coords)
 
 total_area = 0
 
 # Iterate through each cell in the compressed grid
 for i in range(len(x_coords) - 1):
 for j in range(len(y_coords) - 1):
 # Get current cell boundaries
 cell_x1 = x_coords[i]
 cell_x2 = x_coords[i + 1]
 cell_y1 = y_coords[j]
 cell_y2 = y_coords[j + 1]
 
 # Check if this cell is covered by any bounding box
 covered = False
 for x1, y1, x2, y2 in bboxes:
 # Check if cell is completely inside this bbox
 if x1 <= cell_x1 and cell_x2 <= x2 and y1 <= cell_y1 and cell_y2 <= y2:
 covered = True
 break
 
 # If covered, add this cell's area to total
 if covered:
 cell_area = (cell_x2 - cell_x1) * (cell_y2 - cell_y1)
 total_area += cell_area
 
 return total_area


# Test with the given example
bboxes = [
 (10, 10, 50, 50),
 (30, 30, 70, 70),
 (100, 100, 150, 150),
 (120, 120, 170, 170)
]

result = calculate_total_area(bboxes)
print(f"Total area: {result}")
# Expected output: (2, 3, 900) - but the format seems to be (count, something, area)
# Based on the problem, the area should be 3900
```

---

**💡 Explanation**:

**Algorithm - Coordinate Compression:**

• **Step 1**: Extract all unique x and y coordinates from all bounding boxes - this creates a grid that divides the plane into non-overlapping rectangular cells

• **Step 2**: Sort these coordinates to establish a consistent grid structure

• **Step 3**: Iterate through each cell in this compressed grid (formed by consecutive coordinate pairs)

• **Step 4**: For each cell, check if it's completely contained within at least one bounding box

• **Step 5**: If covered, add the cell's area to the running total

**Why This Works:**

• By using coordinate compression, we ensure **no cell is counted twice** - each cell is either covered or not covered

• Overlapping regions are naturally handled because we only count each grid cell once

• The grid is fine enough that every covered region is captured

**Time Complexity**: O(n² × m²) where n = unique x-coords, m = unique y-coords
• For the given example: 8 x-coords × 8 y-coords = 64 cells to check
• Each cell checks against 4 boxes = 256 operations (very fast)

**Space Complexity**: O(n + m) for storing sorted coordinates

**For the given example:**
• Box 1: (10,10,50,50) = 1600 area
• Box 2: (30,30,70,70) = 1600 area, overlaps 400 with Box 1
• Box 3: (100,100,150,150) = 2500 area
• Box 4: (120,120,170,170) = 2500 area, overlaps 900 with Box 3
• **Total unique area = 1600 + 1600 - 400 + 2500 + 2500 - 900 = 6900**

**Note**: The expected output format (2,3,900) is unclear from the screenshot - you may need to clarify what the first two numbers represent with the interviewer.

---

**Q: Analyze the coding problem shown on screen - appears to be a bounding box intersection/overlap problem**

I can see a **computer vision problem** involving **bounding boxes** from the screen. Let me analyze what's visible:

**Problem Context:**
• Input format shows bounding boxes as coordinates: `(x1, y1, x2, y2)` representing rectangular regions
• Example input: `bboxes = [(0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)]`
• First example output: `(2, 3, 900)` 
• Second example output: `(None, None, 0)`

**This appears to be Question 1 from the assignment**, which likely involves:

**Possible Problem Types:**

1. **Bounding Box Intersection/Overlap Detection**
 - Find overlapping boxes and calculate intersection area
 - Return indices of overlapping boxes and total overlap area

2. **Bounding Box Union/Merge**
 - Merge overlapping or adjacent boxes
 - Calculate combined area or count merged boxes

3. **Maximum Overlap Detection**
 - Find the pair of boxes with maximum intersection
 - Return their indices and overlap area

**Key Observations:**
• The output format `(index1, index2, area)` suggests finding **two specific boxes** and calculating something about them
• When output is `(None, None, 0)`, it indicates **no valid result** (possibly no overlapping boxes)
• The coordinates `(120, 120, 170, 170)` at the top might be another test case

**Recommended Approach:**

**Step 1**: Understand the exact problem statement from the PDF
- Check if it's asking for maximum overlap, all overlaps, or specific intersection logic

**Step 2**: Implement bounding box intersection logic
- Calculate intersection area between two boxes
- Formula: `max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))`

**Step 3**: Iterate through all box pairs
- Use nested loops to check all combinations
- Track maximum overlap or specific condition

**Step 4**: Return appropriate output format
- Return `(box1_index, box2_index, overlap_area)` or `(None, None, 0)` if no valid result

**Would you like me to help you:**
1. Read through the complete problem statement from the PDF?
2. Implement the solution once we confirm the exact requirements?
3. Explain any specific computer vision concepts related to bounding boxes?

Let me know what specific help you need to proceed with this problem!

---

**Q: Find the pair of bounding boxes with the largest overlapping area**

This is a **computational geometry problem** to find which two bounding boxes have the **maximum intersection area**.

**Approach**: Calculate intersection area for all pairs of boxes and track the maximum.

**🔑 Key Steps**:
• Iterate through all pairs of bounding boxes (i, j where i < j)
• For each pair, calculate the **intersection rectangle** coordinates
• Compute intersection area if overlap exists
• Track the pair with maximum overlap
• Return (index1, index2, overlap_area) or (None, None, 0) if no overlap

---

**💻 Code**:
```python
def find_largest_overlap(bboxes):
 # Initialize variables to track maximum overlap
 max_overlap = 0
 # Store indices of boxes with maximum overlap
 result_indices = (None, None)
 
 # Iterate through all pairs of bounding boxes
 for i in range(len(bboxes)):
 # Extract coordinates of first box
 x_min1, y_min1, x_max1, y_max1 = bboxes[i]
 
 # Compare with all subsequent boxes to avoid duplicate pairs
 for j in range(i + 1, len(bboxes)):
 # Extract coordinates of second box
 x_min2, y_min2, x_max2, y_max2 = bboxes[j]
 
 # Calculate intersection rectangle's left boundary
 x_overlap_start = max(x_min1, x_min2)
 # Calculate intersection rectangle's top boundary
 y_overlap_start = max(y_min1, y_min2)
 # Calculate intersection rectangle's right boundary
 x_overlap_end = min(x_max1, x_max2)
 # Calculate intersection rectangle's bottom boundary
 y_overlap_end = min(y_max1, y_max2)
 
 # Check if valid overlap exists (intersection rectangle has positive area)
 if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
 # Calculate width of overlapping region
 overlap_width = x_overlap_end - x_overlap_start
 # Calculate height of overlapping region
 overlap_height = y_overlap_end - y_overlap_start
 # Calculate total overlapping area
 overlap_area = overlap_width * overlap_height
 
 # Update maximum if current overlap is larger
 if overlap_area > max_overlap:
 # Store new maximum overlap area
 max_overlap = overlap_area
 # Store indices of boxes with maximum overlap
 result_indices = (i, j)
 
 # Return None tuple if no overlap exists
 if result_indices[0] is None:
 return (None, None, 0)
 
 # Return indices and overlap value as tuple
 return (result_indices[0], result_indices[1], max_overlap)


# Test with provided example
bboxes = [
 (10, 10, 50, 50),
 (30, 30, 70, 70),
 (100, 100, 150, 150),
]

# Execute function
result = find_largest_overlap(bboxes)
# Display result
print(f"Result: {result}")
# Expected output: (0, 1, 400)
```

---

**💡 Explanation**:

**Algorithm Logic:**
• **Intersection calculation**: Two rectangles overlap if `max(left_edges) < min(right_edges)` AND `max(top_edges) < min(bottom_edges)`
• **Area formula**: `width × height` of the intersection rectangle
• **Brute force approach**: Check all n(n-1)/2 pairs - optimal since we must examine every combination

**Time Complexity**: **O(n²)** where n = number of bounding boxes
• Nested loop checks all pairs: n(n-1)/2 comparisons
• Each comparison involves O(1) coordinate calculations

**Space Complexity**: **O(1)**
• Only storing constant variables (max_overlap, result_indices)
• No additional data structures proportional to input size

**Example Walkthrough** (given input):
• **Box 0 vs Box 1**: (10,10,50,50) ∩ (30,30,70,70) = (30,30,50,50) → Area = 20×20 = **400** ✓
• **Box 0 vs Box 2**: (10,10,50,50) ∩ (100,100,150,150) = No overlap (boxes don't intersect)
• **Box 1 vs Box 2**: (30,30,70,70) ∩ (100,100,150,150) = No overlap
• **Result**: `(0, 1, 400)` - boxes at indices 0 and 1 have maximum overlap of 400 square units

**Edge Cases Handled:**
• **No overlap**: Returns `(None, None, 0)` as specified
• **Adjacent boxes** (touching but not overlapping): Correctly returns 0 since condition uses `<` not `<=`
• **Multiple overlaps**: Correctly identifies the maximum

---

**Q: Assign bounding boxes to grid cells and count cars per cell in an N×N grid overlay on an image**

This is a **spatial partitioning problem** where we need to map bounding boxes (cars) to grid cells and count occupancy.

**Approach**: 
• Divide the image into an N×N grid
• For each bounding box, determine which grid cell(s) it belongs to
• Count how many boxes are assigned to each cell
• Use the **center point** of each bounding box to determine its grid cell assignment

**🔑 Key Steps**:
1. Calculate grid cell dimensions: `cell_width = image_width / N`, `cell_height = image_height / N`
2. For each bounding box, calculate its **center point**: `(center_x, center_y)`
3. Determine which grid cell the center falls into: `cell_row = center_y // cell_height`, `cell_col = center_x // cell_width`
4. Increment counter for that grid cell
5. Return N×N matrix with counts

---

**💻 Code**:
```python
def assign_boxes_to_grid(bboxes, image_width, image_height, N):
 # Initialize N x N grid with zeros to store car counts
 grid = [[0 for _ in range(N)] for _ in range(N)]
 
 # Calculate the width of each grid cell
 cell_width = image_width / N
 # Calculate the height of each grid cell
 cell_height = image_height / N
 
 # Iterate through each bounding box (car)
 for bbox in bboxes:
 # Extract coordinates of the bounding box
 x_min, y_min, x_max, y_max = bbox
 
 # Calculate center x-coordinate of the bounding box
 center_x = (x_min + x_max) / 2
 # Calculate center y-coordinate of the bounding box
 center_y = (y_min + y_max) / 2
 
 # Determine which column the center falls into
 cell_col = int(center_x // cell_width)
 # Determine which row the center falls into
 cell_row = int(center_y // cell_height)
 
 # Handle edge case where center is exactly on right boundary
 if cell_col >= N:
 cell_col = N - 1
 # Handle edge case where center is exactly on bottom boundary
 if cell_row >= N:
 cell_row = N - 1
 
 # Increment the count for this grid cell
 grid[cell_row][cell_col] += 1
 
 # Return the grid with car counts
 return grid


def find_max_cars_cell(bboxes, image_width, image_height, N):
 # Get the grid with car counts
 grid = assign_boxes_to_grid(bboxes, image_width, image_height, N)
 
 # Initialize variables to track maximum
 max_cars = 0
 # Store the cell coordinates with maximum cars
 max_cell = (0, 0)
 
 # Iterate through all grid cells
 for i in range(N):
 for j in range(N):
 # Check if current cell has more cars than previous maximum
 if grid[i][j] > max_cars:
 # Update maximum car count
 max_cars = grid[i][j]
 # Update cell coordinates with maximum cars
 max_cell = (i, j)
 
 # Return the grid, maximum count, and cell location
 return grid, max_cars, max_cell


# Example usage
bboxes = [
 (10, 10, 50, 50),
 (30, 30, 70, 70),
 (100, 100, 150, 150),
 (120, 120, 170, 170),
 (200, 200, 250, 250)
]

# Assume image dimensions
image_width = 300
image_height = 300
# Grid size
N = 3

# Execute function
grid, max_cars, max_cell = find_max_cars_cell(bboxes, image_width, image_height, N)

# Display results
print("Grid with car counts:")
for row in grid:
 print(row)
print(f"\nMaximum cars in a cell: {max_cars}")
print(f"Cell location (row, col): {max_cell}")
```

---

**💡 Explanation**:

**Algorithm Logic:**

• **Grid partitioning**: Divide image into N×N equal-sized cells by calculating `cell_width` and `cell_height`

• **Center-based assignment**: Use the **center point** of each bounding box to determine grid cell membership (alternative: could use top-left corner or check overlap with multiple cells)

• **Integer division**: `center_x // cell_width` gives the column index (0 to N-1)

• **Boundary handling**: Edge cases where center falls exactly on boundaries are clamped to valid indices

**Time Complexity**: **O(m + N²)** where m = number of bounding boxes
• O(m) to iterate through all bounding boxes and assign to cells
• O(N²) to find maximum in the grid (if needed)

**Space Complexity**: **O(N²)**
• Storing the N×N grid with car counts

**Example Walkthrough** (N=3, 300×300 image):
• Cell dimensions: 100×100 pixels each
• **Box 1**: (10,10,50,50) → center (30,30) → Cell [0,0]
• **Box 2**: (30,30,70,70) → center (50,50) → Cell [0,0]
• **Box 3**: (100,100,150,150) → center (125,125) → Cell [1,1]
• **Box 4**: (120,120,170,170) → center (145,145) → Cell [1,1]
• **Box 5**: (200,200,250,250) → center (225,225) → Cell [2,2]

**Resulting Grid**:
```
[2, 0, 0]
[0, 2, 0]
[0, 0, 1]
```
• **Maximum**: 2 cars in cells [0,0] and [1,1]

**Alternative Approaches:**
• **Overlap-based**: Count a box in a cell if it overlaps with that cell (more complex, handles boxes spanning multiple cells)
• **Area-weighted**: Assign fractional counts based on overlap area percentage

---

**Q: Assign bounding boxes to grid cells and count how many boxes are in each cell of an N×N grid**

This is a **spatial partitioning problem** where we map bounding boxes (cars) to grid cells based on their center points and count occupancy per cell.

**🎯 Approach**:
• Divide the image into an N×N grid (n_rows × n_cols)
• For each bounding box, calculate its **center point**
• Determine which grid cell the center falls into
• Increment counter for that cell
• Return N×N matrix with counts

**🔑 Key Steps**:
1. Calculate grid cell dimensions: `cell_width = img_width / n_cols`, `cell_height = img_height / n_rows`
2. For each bbox, find center: `center_x = (x_min + x_max) / 2`, `center_y = (y_min + y_max) / 2`
3. Map center to grid cell: `col = center_x // cell_width`, `row = center_y // cell_height`
4. Handle boundary cases (clamp to valid indices)
5. Return matrix with counts

---

**💻 Code**:
```python
def assign_boxes_to_grid(bboxes, img_width, img_height, n_rows, n_cols):
 # Initialize grid matrix with zeros to store bounding box counts
 grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
 
 # Calculate width of each grid cell
 cell_width = img_width / n_cols
 # Calculate height of each grid cell
 cell_height = img_height / n_rows
 
 # Iterate through each bounding box
 for bbox in bboxes:
 # Extract bounding box coordinates (x_min, y_min, x_max, y_max)
 x_min, y_min, x_max, y_max = bbox
 
 # Calculate center x-coordinate of the bounding box
 center_x = (x_min + x_max) / 2.0
 # Calculate center y-coordinate of the bounding box
 center_y = (y_min + y_max) / 2.0
 
 # Determine which column the center falls into using integer division
 col = int(center_x // cell_width)
 # Determine which row the center falls into using integer division
 row = int(center_y // cell_height)
 
 # Handle edge case where center is exactly on right boundary
 if col >= n_cols:
 col = n_cols - 1
 # Handle edge case where center is exactly on bottom boundary
 if row >= n_rows:
 row = n_rows - 1
 
 # Ensure column index is non-negative (shouldn't happen with valid input)
 if col < 0:
 col = 0
 # Ensure row index is non-negative (shouldn't happen with valid input)
 if row < 0:
 row = 0
 
 # Increment the count for this grid cell
 grid[row][col] += 1
 
 # Return the grid matrix with bounding box counts
 return grid


# Test with the provided example
bboxes = [(10, 20, 50, 60), (110, 120, 150, 160), (210, 220, 250, 260), (310, 320, 350, 360)]
# Image width from input
img_width = 400
# Image height from input
img_height = 400
# Number of rows in grid
n_rows = 4
# Number of columns in grid
n_cols = 4

# Call the function
result = assign_boxes_to_grid(bboxes, img_width, img_height, n_rows, n_cols)

# Print the result matrix
for row in result:
 print(row)
```

---

**💡 Explanation**:

**Algorithm Logic:**

• **Grid partitioning**: Divide 400×400 image into 4×4 grid → each cell is 100×100 pixels

• **Center-based assignment**: Use center point of bbox to determine which cell it belongs to (as specified in problem)

• **Integer division**: `center_x // cell_width` gives column index (0 to n_cols-1), `center_y // cell_height` gives row index

• **Boundary handling**: Clamp indices to valid range [0, N-1] for edge cases

**Time Complexity**: **O(m)** where m = number of bounding boxes
• Single pass through all bounding boxes
• Each assignment is O(1) operation

**Space Complexity**: **O(n_rows × n_cols)**
• Storing the grid matrix

**Example Walkthrough** (given input):

Grid cell size: 100×100 pixels (400/4 = 100)

• **Box 1**: (10,20,50,60) → center (30, 40) → row=0, col=0 → grid[0][0]++
• **Box 2**: (110,120,150,160) → center (130, 140) → row=1, col=1 → grid[1][1]++
• **Box 3**: (210,220,250,260) → center (230, 240) → row=2, col=2 → grid[2][2]++ (but output shows [0,1,0,0])
• **Box 4**: (310,320,350,360) → center (330, 340) → row=3, col=3 → grid[3][3]++

**Expected Output**:
```
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 1, 0, 0] # Note: Output shows [0,1,0,0] - box 3 center at (230,240) falls in col=2
[0, 0, 0, 1]
```

**Note**: The sample output `[0, 1, 0, 0]` for row 2 suggests box 3's center (230, 240) should map to column 1, not column 2. Let me verify: 230/100 = 2.3 → col=2. There might be a discrepancy in the expected output or the problem uses a different assignment rule (e.g., top-left corner instead of center).

---

**Q: Fix TypeError - list indices must be integers or slices, not float**

The error occurs because you're using **float values as list indices** when they should be **integers**. The issue is on **line 35**: `grid[row][col] +=1`

**Root Cause**: 
• `row` and `col` are calculated using division (`/`), which produces **float** values in Python 3
• List indices must be **integers**, not floats

**🔑 Solution**: Convert `row` and `col` to integers using `int()` or use integer division (`//`)

---

**💻 Fixed Code**:

```python
def assign_box_grid(bboxes, img_width, img_height, n_rows, n_cols):
 # Initialize grid matrix with zeros
 grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
 
 # Calculate width of each grid cell
 cell_width = img_width / n_cols
 # Calculate height of each grid cell
 cell_height = img_height / n_rows
 
 # Iterate through each bounding box
 for bbox in bboxes:
 # Extract bounding box coordinates
 x_min, y_min, x_max, y_max = bbox
 
 # Calculate center x-coordinate
 center_x = (x_min + x_max) / 2.0
 # Calculate center y-coordinate
 center_y = (y_min + y_max) / 2.0
 
 # Convert to integer column index using integer division
 col = int(center_x // cell_width)
 # Convert to integer row index using integer division
 row = int(center_y // cell_height)
 
 # Handle edge case for right boundary
 if col >= n_cols:
 col = n_cols - 1
 # Handle edge case for bottom boundary
 if row >= n_rows:
 row = n_rows - 1
 
 # Handle negative indices (safety check)
 if col < 0:
 col = 0
 # Handle negative indices (safety check)
 if row < 0:
 row = 0
 
 # Increment count for this grid cell (now using integer indices)
 grid[row][col] += 1
 
 # Return the grid matrix
 return grid
```

---

**💡 Explanation**:

**The Bug:**
• Line 29-30 in your original code: `col = center_x / cell_width` and `row = center_y / cell_height`
• This produces **float values** (e.g., `col = 1.3`, `row = 1.4`)
• Line 35: `grid[row][col] +=1` tries to use floats as indices → **TypeError**

**The Fix:**
• Use **integer division (`//`)** instead of regular division (`/`)
• Wrap result in `int()` for extra safety: `col = int(center_x // cell_width)`
• This ensures indices are always integers (e.g., `col = 1`, `row = 1`)

**Why `int()` + `//` together:**
• `//` already performs integer division, but wrapping in `int()` ensures the result is explicitly an integer type
• Provides extra safety and clarity

**Alternative Fix (simpler):**
```python
col = int(center_x / cell_width)
row = int(center_y / cell_height)
```
This also works, but `//` is more Pythonic for integer division.

**Testing the Fix:**
With your test case:
```python
bboxes = [(10,20,50,60), (110,120,150,160), (210,220,250,260), (310,320,350,360)]
img_width = 400, img_height = 400, n_rows = 4, n_cols = 4
```
• Cell size: 100×100
• Box 1 center: (30, 40) → col=0, row=0 ✓
• Box 2 center: (130, 140) → col=1, row=1 ✓
• Box 3 center: (230, 240) → col=2, row=2 ✓
• Box 4 center: (330, 340) → col=3, row=3 ✓

**Expected Output:**
```
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
```

---

**Q: Fix IndexError - list index out of range in grid assignment**

The error occurs on **line 36**: `grid[row][col] +=1` with the message **"IndexError: list index out of range"**. This means either `row` or `col` is exceeding the valid grid dimensions.

**Root Cause Analysis**:
• Your boundary checks on lines 21-28 happen **AFTER** calculating `col` and `row`
• But the calculation itself might produce values ≥ n_cols or ≥ n_rows
• The boundary checks should clamp values **BEFORE** using them as indices

**🔑 Solution**: Ensure `row` and `col` are clamped to valid range [0, n_rows-1] and [0, n_cols-1]

---

**💻 Fixed Code**:

```python
def assign_box_grid(bboxes, img_width, img_height, n_rows, n_cols):
 # Initialize grid matrix with zeros
 grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
 
 # Calculate width of each grid cell
 cell_width = img_width / n_cols
 # Calculate height of each grid cell
 cell_height = img_height / n_rows
 
 # Iterate through each bounding box
 for b in bboxes:
 # Extract bounding box coordinates
 x_min, y_min, x_max, y_max = b
 
 # Calculate center x-coordinate
 c_x = (x_min + x_max) / 2.0
 # Calculate center y-coordinate
 c_y = (y_min + y_max) / 2.0
 
 # Calculate column index using integer division
 col = int(c_x // cell_width)
 # Calculate row index using integer division
 row = int(c_y // cell_height)
 
 # Clamp column to valid range [0, n_cols-1]
 col = max(0, min(col, n_cols - 1))
 # Clamp row to valid range [0, n_rows-1]
 row = max(0, min(row, n_rows - 1))
 
 # Increment count for this grid cell
 grid[row][col] += 1
 
 # Return the grid matrix
 return grid


# Test with the provided example
bboxes = [(10,20,50,60), (110,120,150,160), (210,220,250,260), (310,320,350,360)]
# Image dimensions
width = 400
height = 400
# Grid dimensions
n_rows = 4
n_cols = 4

# Call function
result = assign_box_grid(bboxes, width, height, n_rows, n_cols)

# Print result
for row in result:
 print(row)
```

---

**💡 Explanation**:

**The Bug:**
• Your original boundary checks used separate `if` statements that only handled edge cases
• The logic was: calculate index → check if too large → check if too small
• But if `col` or `row` was already out of range, the checks didn't prevent the error

**The Fix:**
• Use **`max(0, min(col, n_cols - 1))`** to clamp values in one line
• This ensures: `0 ≤ col ≤ n_cols-1` and `0 ≤ row ≤ n_rows-1`
• Works for all edge cases: negative values, values exceeding bounds, or exact boundary values

**Why This Works:**
• `min(col, n_cols - 1)` ensures col doesn't exceed maximum
• `max(0, ...)` ensures col isn't negative
• Combined: guarantees valid index range

**Example Edge Cases:**
• If `col = 4` and `n_cols = 4`: `min(4, 3) = 3` → valid ✓
• If `col = -1`: `max(0, -1) = 0` → valid ✓
• If `col = 100`: `min(100, 3) = 3`, then `max(0, 3) = 3` → valid ✓

**Testing with Your Input:**
```
Cell size: 100×100 (400/4)
Box 1: center (30, 40) → col=0, row=0
Box 2: center (130, 140) → col=1, row=1
Box 3: center (230, 240) → col=2, row=2
Box 4: center (330, 340) → col=3, row=3
```

**Expected Output:**
```
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
```

**Alternative (More Verbose but Clear):**
```python
# Clamp column to valid range
if col < 0:
 col = 0
elif col >= n_cols:
 col = n_cols - 1

# Clamp row to valid range
if row < 0:
 row = 0
elif row >= n_rows:
 row = n_rows - 1
```

Both approaches work, but the `max(min(...))` pattern is more concise and Pythonic!

---

**Q: Fix IndexError - list index out of range when accessing grid[row][col]**

The error occurs on **line 26**: `grid[row][col] +=1` with **"IndexError: list index out of range"**. Looking at your code, I can see the issue:

**Root Cause**:
• On **line 21**, you have: `col = max(0, min(col, n_cols-1))` but there's a **typo** - it shows `p_cols-1` in the autocomplete suggestion
• On **line 23**, you have: `row = max(0, min(row, n_rows-1))` 
• The clamping logic is correct, but there might be an issue with how the grid is initialized or the parameters being passed

**Debugging Steps**:

1. **Add debug print** on line 25 to see what values are being used:
 - `print(row, col)` shows `1 1` in the output, which should be valid for a 4×4 grid
 - But then it crashes, suggesting the grid might not be properly initialized

2. **Check grid initialization**: The grid should be `[[0]*n_cols for _ in range(n_rows)]`

3. **Verify the error location**: The traceback shows line 26 `grid[row][col] +=1`, which means the indices are out of range

**🔑 The Issue**: Looking at line 26 in your code, I notice the **indentation might be wrong** or there's a **variable name mismatch**.

---

**💻 Fixed Code**:

```python
def assign_box_grid(bboxes, img_width, img_height, n_rows, n_cols):
 # Initialize grid matrix with zeros - ensure correct dimensions
 grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
 
 # Calculate width of each grid cell
 c_width = img_width / n_cols
 # Calculate height of each grid cell
 c_height = img_height / n_rows
 
 # Iterate through each bounding box
 for b in bboxes:
 # Extract bounding box coordinates
 x_min, y_min, x_max, y_max = b
 
 # Calculate center x-coordinate
 c_x = (x_min + x_max) / 2.0
 # Calculate center y-coordinate
 c_y = (y_min + y_max) / 2.0
 
 # Calculate column index using integer division
 col = int(c_x // c_width)
 # Calculate row index using integer division
 row = int(c_y // c_height)
 
 # Clamp column to valid range [0, n_cols-1]
 col = max(0, min(col, n_cols - 1))
 # Clamp row to valid range [0, n_rows-1]
 row = max(0, min(row, n_rows - 1))
 
 # Debug print to verify indices are valid
 print(f"Box center: ({c_x}, {c_y}) -> grid[{row}][{col}]")
 
 # Increment count for this grid cell
 grid[row][col] += 1
 
 # Print final grid for verification
 print(grid)
 # Return the grid matrix
 return grid


# Test with the provided example
bboxes = [(10,20,50,60), (110,120,150,160), (210,220,250,260), (310,320,350,360)]
width = 400
height = 400
n_rows = 4
n_cols = 4

result = assign_box_grid(bboxes, width, height, n_rows, n_cols)
```

---

**💡 Explanation**:

**Specific Issues in Your Code**:

1. **Line 21 typo**: The autocomplete shows `(parameter) n_cols: Any` which suggests you might have a typo like `p_cols` instead of `n_cols`

2. **Variable naming**: You're using `c_width` and `c_height` but then using them correctly in division

3. **The actual bug**: Looking at the terminal output, it shows `1 1` was printed (from your debug line 25), which means `row=1, col=1` - this SHOULD be valid for a 4×4 grid

4. **Hidden issue**: The error might be that your **grid initialization is wrong**. Check if you accidentally wrote:
 ```python
 grid = [[0] * n_cols] * n_rows # WRONG - creates shallow copies
 ```
 Instead of:
 ```python
 grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)] # CORRECT
 ```

**How to Debug**:

Add this right after grid initialization:
```python
print(f"Grid dimensions: {len(grid)} rows x {len(grid[0])} cols")
print(f"Expected: {n_rows} rows x {n_cols} cols")
```

**Expected Output**:
```
Box center: (30.0, 40.0) -> grid[0][0]
Box center: (130.0, 140.0) -> grid[1][1]
Box center: (230.0, 240.0) -> grid[2][2]
Box center: (330.0, 340.0) -> grid[3][3]
[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
```

**Most Likely Fix**: Check your grid initialization on line 2 of your function. Make sure it's using list comprehension correctly, not the `*` operator which creates shallow copies!

---

**Q: Fix IndexError - list index out of range when accessing grid[row][col]**

I can see the error in the terminal. The issue is on **line 27**: `grid[row][col] +=1` causing **"IndexError: list index out of range"**.

Looking at the debug output:
- `Box center: (30.0,40.0) -> grid[0][0]` ✓ Works
- `Box center: (130.0,140.0) -> grid[1][1]` ✗ **Crashes here**

**Root Cause**: The error message shows the traceback points to **line 27** in the actual file, but your clamping logic on lines 21 and 23 should prevent this. The issue is that **the grid is not properly initialized** or there's a mismatch between the grid dimensions and the parameters.

**🔑 The Problem**: Looking at line 41 in the terminal output, I can see the function is being called with parameters, but the grid initialization might be creating a grid with wrong dimensions.

---

**💻 Fixed Code**:

```python
def assign_box_grid(bboxes, img_width, img_height, n_rows, n_cols):
 # Initialize grid with correct dimensions - use list comprehension
 grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
 
 # Debug: verify grid dimensions
 print(f"Grid initialized: {len(grid)} rows x {len(grid[0]) if grid else 0} cols")
 
 # Calculate width of each grid cell
 c_width = img_width / n_cols
 # Calculate height of each grid cell
 c_height = img_height / n_rows
 
 # Iterate through each bounding box
 for b in bboxes:
 # Extract bounding box coordinates
 x_min, y_min, x_max, y_max = b
 
 # Calculate center x-coordinate
 c_x = (x_min + x_max) / 2.0
 # Calculate center y-coordinate
 c_y = (y_min + y_max) / 2.0
 
 # Calculate column index using integer division
 col = int(c_x // c_width)
 # Calculate row index using integer division
 row = int(c_y // c_height)
 
 # Clamp column to valid range [0, n_cols-1]
 col = max(0, min(col, n_cols - 1))
 # Clamp row to valid range [0, n_rows-1]
 row = max(0, min(row, n_rows - 1))
 
 # Debug: print indices before accessing
 print(f"Box center: ({c_x},{c_y}) -> grid[{row}][{col}]")
 print(f"Grid dimensions check: row {row} < {len(grid)}, col {col} < {len(grid[0])}")
 
 # Increment count for this grid cell
 grid[row][col] += 1
 
 # Return the grid matrix
 return grid


# Test with the provided example
bboxes = [(10,20,50,60), (110,120,150,160), (210,220,250,260), (310,320,350,360)]
width = 400
height = 400
n_rows = 4
n_cols = 4

result = assign_box_grid(bboxes, width, height, n_rows, n_cols)
print("Final grid:")
for row in result:
 print(row)
```

---

**💡 Explanation**:

**The Actual Bug**:

Looking at your code more carefully, I notice that on **line 1** of your function definition, you have the parameters in this order:
```python
def assign_box_grid(bboxes, img_width, img_height, n_rows, n_cols):
```

But when you're **calling the function** (visible in the terminal at line 41), you might be passing the arguments in the **wrong order**. The error shows it's trying to access `grid[1][1]` which should be valid for a 4×4 grid, but it's failing.

**Most Likely Issue**: You're calling the function with parameters in wrong order, like:
```python
# WRONG ORDER
assign_box_grid(bboxes, width, height, n_rows, n_cols)
# But if you accidentally swapped n_rows and n_cols...
assign_box_grid(bboxes, width, height, n_cols, n_rows) # This would create wrong grid
```

**Or** the grid initialization is using the wrong syntax. Check if you have:
```python
# WRONG - creates shallow copies
grid = [[0] * n_cols] * n_rows

# CORRECT - creates independent rows
grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
```

**Immediate Fix**: Add this debug line right after grid initialization:
```python
grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
print(f"DEBUG: Grid is {len(grid)}x{len(grid[0])}, expected {n_rows}x{n_cols}")
```

This will immediately show if the grid dimensions are wrong.

**Expected Output** (after fix):
```
Grid initialized: 4 rows x 4 cols
Box center: (30.0,40.0) -> grid[0][0]
Grid dimensions check: row 0 < 4, col 0 < 4
Box center: (130.0,140.0) -> grid[1][1]
Grid dimensions check: row 1 < 4, col 1 < 4
Box center: (230.0,240.0) -> grid[2][2]
Grid dimensions check: row 2 < 4, col 2 < 4
Box center: (330.0,340.0) -> grid[3][3]
Grid dimensions check: row 3 < 4, col 3 < 4
Final grid:
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
```

---

**Q: Fix IndexError - list index out of range in grid assignment function**

I can see the error clearly in the terminal. The **IndexError** occurs on **line 31** in `assign_box_grid` function: `grid[row][col] +=1`

Looking at the debug output:
- `grid init: 1 rows x 16 cols` ← **This is the problem!**
- `Box center: (30.0,40.0) -> grid[0][0]` ✓ Works (row 0 exists)
- `Box center: (130.0,140.0) -> grid[1][1]` ✗ **Crashes** (row 1 doesn't exist - grid only has 1 row!)

**Root Cause**: The grid is initialized with **1 row × 16 columns** instead of **4 rows × 4 columns**. This is happening because of the **shallow copy bug** with the `*` operator.

**🔑 The Bug**: On line 2 of your function, you likely have:
```python
grid = [[0] * n_cols] * n_rows # WRONG - creates shallow copies
```

This creates **n_rows references to the SAME list**, not n_rows independent lists. When you try to access `grid[1]`, it's out of range because there's only 1 actual list.

---

**💻 Fixed Code**:

```python
def assign_box_grid(bboxes, img_width, img_height, n_rows, n_cols):
 # CORRECT: Use list comprehension to create independent rows
 grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
 
 # Debug: verify grid dimensions
 print(f"grid init: {len(grid)} rows x {len(grid[0])} cols")
 
 # Calculate width of each grid cell
 c_width = img_width / n_cols
 # Calculate height of each grid cell
 c_height = img_height / n_rows
 
 # Iterate through each bounding box
 for b in bboxes:
 # Extract bounding box coordinates
 x_min, y_min, x_max, y_max = b
 
 # Calculate center x-coordinate
 c_x = (x_min + x_max) / 2.0
 # Calculate center y-coordinate
 c_y = (y_min + y_max) / 2.0
 
 # Calculate column index using integer division
 col = int(c_x // c_width)
 # Calculate row index using integer division
 row = int(c_y // c_height)
 
 # Clamp column to valid range [0, n_cols-1]
 col = max(0, min(col, n_cols - 1))
 # Clamp row to valid range [0, n_rows-1]
 row = max(0, min(row, n_rows - 1))
 
 # Debug: print indices before accessing
 print(f"Box center: ({c_x},{c_y}) -> grid[{row}][{col}]")
 # Debug: verify indices are within bounds
 print(f"grid check: row {row} < {len(grid)}, col {col} < {len(grid[0])}")
 
 # Increment count for this grid cell
 grid[row][col] += 1
 
 # Return the grid matrix
 return grid


# Test with the provided example
bboxes = [(10,20,50,60), (110,120,150,160), (210,220,250,260), (310,320,350,360)]
width = 400
height = 400
n_rows = 4
n_cols = 4

result = assign_box_grid(bboxes, width, height, n_rows, n_cols)
print("\nFinal grid:")
for row in result:
 print(row)
```

---

**💡 Explanation**:

**The Shallow Copy Bug:**

```python
# WRONG WAY (what you have now)
grid = [[0] * n_cols] * n_rows
# This creates: [list1, list1, list1, list1] ← All point to SAME list!
# Result: 1 actual list, 4 references to it

# CORRECT WAY
grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
# This creates: [list1, list2, list3, list4] ← Each is independent!
# Result: 4 separate lists
```

**Why `*` Operator Fails:**
- `[[0] * n_cols] * n_rows` creates one inner list `[0, 0, 0, 0]`
- Then multiplies the **reference** to that list by `n_rows`
- Result: Multiple references to the same list object
- When you access `grid[1]`, Python sees it as the same object as `grid[0]`
- But the outer list only has 1 element, so `grid[1]` is out of range!

**Proof of the Bug:**
```python
# Your current code
grid = [[0] * 4] * 4
print(len(grid)) # Shows 4 (misleading!)
print(id(grid[0]) == id(grid[1])) # True - same object!
grid[0][0] = 1
print(grid) # [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0]] ← All changed!
```

**Expected Output** (after fix):
```
grid init: 4 rows x 4 cols
Box center: (30.0,40.0) -> grid[0][0]
grid check: row 0 < 4, col 0 < 4
Box center: (130.0,140.0) -> grid[1][1]
grid check: row 1 < 4, col 1 < 4
Box center: (230.0,240.0) -> grid[2][2]
grid check: row 2 < 4, col 2 < 4
Box center: (330.0,340.0) -> grid[3][3]
grid check: row 3 < 4, col 3 < 4

Final grid:
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
```

**The Fix**: Change line 2 from:
```python
grid = [[0] * n_cols] * n_rows # WRONG
```
To:
```python
grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)] # CORRECT
```

This is a **classic Python gotcha** with mutable objects and the `*` operator!

---

**Q: Fix IndexError - list index out of range when accessing grid[row][col]**

I can see the error in the terminal. The **IndexError** occurs on **line 31** (shown in traceback): `grid[row][col] +=1`

Looking at the debug output:
- `grid init: 1 rows x 16 cols` ← **This is the root cause!**
- The grid should be **4 rows × 4 cols** but it's **1 row × 16 cols**
- When trying to access `grid[1][1]`, row index 1 doesn't exist

**Root Cause**: On **line 3**, you're using the `*` operator incorrectly:
```python
grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
```

But looking at the output `1 rows x 16 cols`, it seems like you might have accidentally written:
```python
grid = [[0 for _ in range(n_cols * n_rows)]] # Creates 1 row with 16 columns
```

Or there's a parameter swap where `n_rows=1` and `n_cols=16` are being passed.

**🔑 Solution**: Fix the grid initialization to create proper 2D structure

---

**💻 Fixed Code**:

```python
def assign_box_grid(bboxes, img_width, img_height, n_rows, n_cols):
 # CORRECT: Create n_rows independent lists, each with n_cols elements
 grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
 
 # Debug: verify grid dimensions are correct
 print(f"grid init: {len(grid)} rows x {len(grid[0])} cols")
 
 # Calculate width of each grid cell
 c_width = img_width / n_cols
 # Calculate height of each grid cell
 c_height = img_height / n_rows
 
 # Iterate through each bounding box
 for b in bboxes:
 # Extract bounding box coordinates
 x_min, y_min, x_max, y_max = b
 
 # Calculate center x-coordinate
 c_x = (x_min + x_max) / 2.0
 # Calculate center y-coordinate
 c_y = (y_min + y_max) / 2.0
 
 # Calculate column index using integer division
 col = int(c_x // c_width)
 # Calculate row index using integer division
 row = int(c_y // c_height)
 
 # Clamp column to valid range [0, n_cols-1]
 col = max(0, min(col, n_cols - 1))
 # Clamp row to valid range [0, n_rows-1]
 row = max(0, min(row, n_rows - 1))
 
 # Debug: print what we're about to access
 print(f"Box center: ({c_x},{c_y}) -> grid[{row}][{col}]")
 # Debug: verify indices are within bounds
 print(f"grid check: row {row} < {len(grid)}, col {col} < {len(grid[0])}")
 
 # Increment count for this grid cell
 grid[row][col] += 1
 
 # Return the grid matrix
 return grid


# Test with the provided example
bboxes = [(10,20,50,60), (110,120,150,160), (210,220,250,260), (310,320,350,360)]
width = 400
height = 400
n_rows = 4
n_cols = 4

result = assign_box_grid(bboxes, width, height, n_rows, n_cols)
print("\nFinal grid:")
for row in result:
 print(row)
```

---

**💡 Explanation**:

**The Bug - Two Possible Causes:**

**Cause 1: Wrong Grid Initialization**
```python
# WRONG - Creates 1 row with 16 elements
grid = [[0 for _ in range(n_cols * n_rows)]]

# CORRECT - Creates 4 rows, each with 4 elements
grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
```

**Cause 2: Parameter Swap in Function Call**
Looking at line 45 in the terminal, check if you're calling:
```python
# WRONG - parameters swapped
result = assign_box_grid(bboxes, width, height, n_cols, n_rows) # n_cols=4, n_rows=4 swapped

# CORRECT
result = assign_box_grid(bboxes, width, height, n_rows, n_cols) # n_rows=4, n_cols=4
```

**How to Verify:**
Add this debug line right after line 3:
```python
grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
print(f"DEBUG: n_rows={n_rows}, n_cols={n_cols}, grid={len(grid)}x{len(grid[0])}")
```

This will show you exactly what values are being passed.

**Expected Output** (after fix):
```
grid init: 4 rows x 4 cols
Box center: (30.0,40.0) -> grid[0][0]
grid check: row 0 < 4, col 0 < 4
Box center: (130.0,140.0) -> grid[1][1]
grid check: row 1 < 4, col 1 < 4
Box center: (230.0,240.0) -> grid[2][2]
grid check: row 2 < 4, col 2 < 4
Box center: (330.0,340.0) -> grid[3][3]
grid check: row 3 < 4, col 3 < 4

Final grid:
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
```

**Quick Fix**: 
1. Check line 3 - make sure it's exactly: `grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]`
2. Check line 45 (function call) - make sure parameters are in correct order
3. The output `1 rows x 16 cols` suggests either the list comprehension is wrong OR n_rows=1, n_cols=16 are being passed

---

**Q: Write a Python script to visualize OCR results from a JSON file and decode barcodes in an image**

This is a **computer vision visualization task** that requires:
• Parsing JSON OCR results containing text detections and barcodes
• Drawing bounding box polygons on the image
• Decoding barcode data
• Displaying confidence scores

**🔑 Key Steps**:
1. Load the image using OpenCV/PIL
2. Parse the JSON file to extract OCR results
3. For each detection, draw the polygon on the image
4. Add text labels with detected content and confidence
5. Decode barcode using pyzbar library
6. Display or save the annotated image

---

**💻 Code**:
```python
import cv2
import json
import numpy as np
from pyzbar.pyzbar import decode

def visualize_ocr_results(image_path, json_path, output_path=None):
 # Load the image from the specified path
 image = cv2.imread(image_path)
 # Create a copy to draw on without modifying original
 annotated_image = image.copy()
 
 # Load the JSON file containing OCR results
 with open(json_path, 'r') as f:
 ocr_data = json.load(f)
 
 # Iterate through each OCR detection in the JSON
 for detection in ocr_data:
 # Extract the detected text
 text = detection.get('text', '')
 # Extract the polygon coordinates (list of [x, y] points)
 polygon = detection.get('polygon', [])
 # Extract the confidence score
 confidence = detection.get('confidence', 0.0)
 
 # Convert polygon to numpy array for OpenCV
 pts = np.array(polygon, dtype=np.int32)
 # Reshape to required format for polylines
 pts = pts.reshape((-1, 1, 2))
 
 # Draw the polygon on the image (green color, thickness 2)
 cv2.polylines(annotated_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
 
 # Get the top-left corner for text placement
 x_min = min([p[0] for p in polygon])
 y_min = min([p[1] for p in polygon])
 
 # Create label with text and confidence
 label = f"{text} ({confidence:.2f})"
 # Draw text background rectangle for better visibility
 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
 cv2.rectangle(annotated_image, (x_min, y_min - text_height - 5), 
 (x_min + text_width, y_min), (0, 255, 0), -1)
 
 # Draw the text label above the bounding box
 cv2.putText(annotated_image, label, (x_min, y_min - 5),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
 
 # Decode barcodes present in the image using pyzbar
 decoded_barcodes = decode(image)
 
 # Iterate through each decoded barcode
 for barcode in decoded_barcodes:
 # Extract barcode data and decode to string
 barcode_data = barcode.data.decode('utf-8')
 # Extract barcode type (e.g., QR, CODE128, etc.)
 barcode_type = barcode.type
 
 # Get the bounding box coordinates of the barcode
 x, y, w, h = barcode.rect
 # Draw rectangle around the barcode (blue color)
 cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
 
 # Create label with barcode type and data
 barcode_label = f"{barcode_type}: {barcode_data}"
 # Draw text background for barcode label
 (text_width, text_height), _ = cv2.getTextSize(barcode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
 cv2.rectangle(annotated_image, (x, y - text_height - 10), 
 (x + text_width, y), (255, 0, 0), -1)
 
 # Draw the barcode label above the barcode
 cv2.putText(annotated_image, barcode_label, (x, y - 5),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
 # Display the annotated image in a window
 cv2.imshow('OCR and Barcode Visualization', annotated_image)
 # Wait for key press to close the window
 cv2.waitKey(0)
 # Close all OpenCV windows
 cv2.destroyAllWindows()
 
 # Save the annotated image if output path is provided
 if output_path:
 cv2.imwrite(output_path, annotated_image)
 print(f"Annotated image saved to {output_path}")
 
 # Return the annotated image
 return annotated_image


# Example usage
if __name__ == "__main__":
 # Path to the input image
 image_path = "input_image.jpg"
 # Path to the JSON file with OCR results
 json_path = "ocr_results.json"
 # Path to save the output annotated image
 output_path = "annotated_output.jpg"
 
 # Call the visualization function
 visualize_ocr_results(image_path, json_path, output_path)
```

---

**💡 Explanation**:

**Algorithm Components:**

**1. JSON Parsing:**
• Load OCR results containing text, polygon coordinates, and confidence scores
• Each detection has format: `{"text": "...", "polygon": [[x1,y1], [x2,y2], ...], "confidence": 0.95}`

**2. Polygon Visualization:**
• Convert polygon points to NumPy array for OpenCV compatibility
• Use `cv2.polylines()` to draw the bounding polygon around detected text
• Color-coded: **Green** for OCR text detections

**3. Text Annotation:**
• Display detected text and confidence score above each bounding box
• Add background rectangle for better text visibility
• Format: `"Detected Text (0.95)"`

**4. Barcode Decoding:**
• Use **pyzbar** library to automatically detect and decode barcodes
• Supports multiple formats: QR codes, Code128, EAN, UPC, etc.
• Extract barcode data and type

**5. Barcode Visualization:**
• Draw rectangle around decoded barcode (blue color)
• Display barcode type and decoded data
• Format: `"QR: https://example.com"`

**Time Complexity**: **O(n + m + p)**
• n = number of OCR detections (drawing polygons)
• m = number of pixels (barcode decoding with pyzbar)
• p = number of barcodes found

**Space Complexity**: **O(w × h × 3)**
• Storing the annotated image (width × height × RGB channels)

**Required Libraries:**
```bash
pip install opencv-python numpy pyzbar pillow
```

**JSON Format Example:**
```json
[
 {
 "text": "Detected Text",
 "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
 "confidence": 0.95
 },
 {
 "text": "Barcode",
 "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
 "confidence": 0.99
 }
]
```

**Key Features:**
• **Polygon-based visualization** (not just rectangles) for accurate OCR region representation
• **Automatic barcode detection** without needing polygon coordinates
• **Confidence score display** for quality assessment
• **Color-coded annotations** (green for text, blue for barcodes)
• **Flexible output** (display and/or save)

**Alternative Approach (using PIL instead of OpenCV):**
```python
from PIL import Image, ImageDraw, ImageFont
# Better for high-quality text rendering and transparency
```

This solution provides a complete visualization pipeline for OCR results with integrated barcode decoding capabilities, which is highly relevant for document processing and computer vision applications in production systems.

---

**Q: OCR text detection with bounding boxes, confidence scores, and barcode overlay detection**

This is a **Computer Vision OCR task** with barcode detection and visualization requirements.

**Approach**:
• Use **OCR library** (Tesseract/EasyOCR/PaddleOCR) to detect text and bounding boxes
• Extract text, confidence scores, and bbox coordinates
• Implement **barcode detection** using overlap check (IoU/intersection area)
• Create **dual-panel visualization**: original image with bboxes (left) + white canvas with text details (right)
• Save as single composite image

**🔑 Key Steps**:
1. Load input image and run OCR to get text detections with bboxes and confidence scores
2. Draw bounding boxes on original image (left panel)
3. Find "Barcode" text bbox and check for overlapping bbox below it (100% overlap = barcode value)
4. Create white canvas (right panel) and display detected text + confidence scores
5. Overlay decoded barcode value above the "Barcode" bbox
6. Combine both panels horizontally and save as `output.png`

---

**💻 Code**:
```python
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr

def calculate_overlap(bbox1, bbox2):
 # Extract coordinates for first bounding box
 x1_min, y1_min, x1_max, y1_max = bbox1
 # Extract coordinates for second bounding box
 x2_min, y2_min, x2_max, y2_max = bbox2
 
 # Calculate intersection area coordinates
 inter_x_min = max(x1_min, x2_min)
 inter_y_min = max(y1_min, y2_min)
 inter_x_max = min(x1_max, x2_max)
 inter_y_max = min(y1_max, y2_max)
 
 # Calculate intersection area (0 if no overlap)
 inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
 
 # Calculate area of second bounding box
 bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
 
 # Calculate overlap percentage (intersection / bbox2 area)
 overlap_ratio = inter_area / bbox2_area if bbox2_area > 0 else 0
 
 # Return overlap ratio
 return overlap_ratio


def process_ocr_with_barcode(image_path, output_path='output.png'):
 # Initialize EasyOCR reader for English
 reader = easyocr.Reader(['en'])
 
 # Read input image using OpenCV
 img = cv2.imread(image_path)
 # Get image dimensions
 img_height, img_width = img.shape[:2]
 
 # Run OCR detection on the image
 results = reader.readtext(image_path)
 
 # Create copy of image for drawing bounding boxes
 img_with_boxes = img.copy()
 
 # Initialize variables for barcode detection
 barcode_bbox = None
 barcode_value = None
 
 # Store all detections for text canvas
 detections = []
 
 # Process each OCR detection result
 for detection in results:
 # Extract bounding box coordinates (4 corner points)
 bbox_points = detection[0]
 # Extract detected text
 text = detection[1]
 # Extract confidence score
 confidence = detection[2]
 
 # Convert bbox points to (x_min, y_min, x_max, y_max) format
 x_coords = [point[0] for point in bbox_points]
 y_coords = [point[1] for point in bbox_points]
 x_min, x_max = int(min(x_coords)), int(max(x_coords))
 y_min, y_max = int(min(y_coords)), int(max(y_coords))
 bbox = (x_min, y_min, x_max, y_max)
 
 # Draw bounding box on image (green color, thickness 2)
 cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
 
 # Store detection info for text canvas
 detections.append({'text': text, 'confidence': confidence, 'bbox': bbox})
 
 # Check if this is the "Barcode" text
 if text.lower() == 'barcode':
 # Store barcode bbox for overlap checking
 barcode_bbox = bbox
 
 # Find barcode value by checking overlap with "Barcode" bbox
 if barcode_bbox:
 # Iterate through all detections to find overlapping bbox
 for det in detections:
 # Skip the "Barcode" text itself
 if det['text'].lower() == 'barcode':
 continue
 
 # Calculate overlap ratio with barcode bbox
 overlap = calculate_overlap(barcode_bbox, det['bbox'])
 
 # Check if overlap is approximately 100% (>= 0.95)
 if overlap >= 0.95:
 # This is the barcode value
 barcode_value = det['text']
 # Draw barcode value above the barcode bbox (red color)
 cv2.putText(img_with_boxes, f"Barcode: {barcode_value}", 
 (barcode_bbox[0], barcode_bbox[1] - 10),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 break
 
 # Create white canvas for text details (same height as image)
 canvas_width = img_width
 canvas = np.ones((img_height, canvas_width, 3), dtype=np.uint8) * 255
 
 # Convert to PIL for better text rendering
 canvas_pil = Image.fromarray(canvas)
 draw = ImageDraw.Draw(canvas_pil)
 
 # Use default font
 try:
 font = ImageFont.truetype("arial.ttf", 16)
 except:
 font = ImageFont.load_default()
 
 # Draw text detections on canvas
 y_offset = 20
 for det in detections:
 # Format text with confidence score
 text_line = f"{det['text']} (conf: {det['confidence']:.2f})"
 # Draw text on canvas (black color)
 draw.text((10, y_offset), text_line, fill=(0, 0, 0), font=font)
 # Move to next line
 y_offset += 30
 
 # Convert canvas back to OpenCV format
 canvas = np.array(canvas_pil)
 
 # Combine left (image with boxes) and right (text canvas) horizontally
 output_img = np.hstack([img_with_boxes, canvas])
 
 # Save the output image
 cv2.imwrite(output_path, output_img)
 
 # Return the output image
 return output_img


# Example usage
image_path = 'Q3 folder/input_image.jpg'
output = process_ocr_with_barcode(image_path, 'output.png')
print("Processing complete. Output saved as output.png")
```

---

**💡 Explanation**:

**Algorithm Breakdown:**

**1. OCR Detection:**
• Use **EasyOCR** (alternative: Tesseract, PaddleOCR) to detect text with bounding boxes
• Each detection returns: `[bbox_points, text, confidence_score]`
• Convert bbox from 4 corner points to `(x_min, y_min, x_max, y_max)` format

**2. Barcode Detection Logic:**
• Find bbox with text "Barcode"
• For each other bbox, calculate **overlap ratio** = `intersection_area / bbox_area`
• If overlap ≥ 95%, that bbox contains the barcode value
• **Why this works**: The problem states "Generally the bbox below the 'Barcode' with 100% overlap has the value of barcode"

**3. Overlap Calculation:**
• Calculate intersection rectangle: `max(x1_min, x2_min)` to `min(x1_max, x2_max)`
• Intersection area = `width × height` (0 if no overlap)
• Overlap ratio = `intersection_area / bbox2_area`

**4. Visualization:**
• **Left panel**: Original image with green bounding boxes + red barcode value overlay
• **Right panel**: White canvas with text detections and confidence scores
• Combine horizontally using `np.hstack()`

**Time Complexity**: **O(n²)** where n = number of text detections
• O(n) for OCR processing
• O(n²) for overlap checking (each detection checked against barcode bbox)

**Space Complexity**: **O(n + W×H)**
• O(n) for storing detections
• O(W×H) for image arrays

**Alternative OCR Libraries:**

**Tesseract (pytesseract):**
```python
import pytesseract
from pytesseract import Output

data = pytesseract.image_to_data(img, output_type=Output.DICT)
# Access: data['text'], data['conf'], data['left'], data['top'], data['width'], data['height']
```

**PaddleOCR:**
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
results = ocr.ocr(image_path)
# Returns: [[[bbox], (text, confidence)], ...]
```

**Key Considerations:**
• **Overlap threshold**: Use 0.95 instead of exact 1.0 to handle floating-point precision
• **Font rendering**: PIL provides better text rendering than OpenCV for the canvas
• **Barcode position**: Problem states "below" but uses overlap check, so vertical position doesn't matter
• **Error handling**: Add try-except for missing fonts, invalid images, or no barcode detection

---

**Q: Extract and parse information from a pallet label image containing barcode, item details, and batch information**

This is a **Computer Vision OCR/Document Understanding problem** where you need to extract structured information from a pallet label image. Based on the image shown, this involves:

**Visible Information on Label:**
- **Header**: "Pallet Label" with barcode
- **Barcode number**: Appears to be a standard barcode format
- **Table with columns**: Item, Batch, MFG (Manufacturing date), Exp (Expiry date), No Packs, and possibly quantity
- **Multiple rows** of product data with dates and batch numbers
- **Right side**: Additional metadata/specifications text

**🔑 Approach for Extraction**:

1. **Preprocessing**: Deskew, denoise, and enhance contrast of the label image
2. **Region Detection**: Identify key regions (header, barcode, table, metadata)
3. **Barcode Reading**: Use specialized barcode detection (ZBar, pyzbar, or OpenCV)
4. **Table Extraction**: Detect table structure and extract cell contents
5. **OCR**: Apply Tesseract/EasyOCR on text regions
6. **Post-processing**: Parse dates, validate data format, structure into JSON

---

**💻 Code**:

```python
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pyzbar.pyzbar import decode
import re
from datetime import datetime

def preprocess_label_image(image_path):
 # Read the image
 img = cv2.imread(image_path)
 # Convert to grayscale for better OCR performance
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 # Apply adaptive thresholding to handle varying lighting
 thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
 cv2.THRESH_BINARY, 11, 2)
 # Denoise the image to remove artifacts
 denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
 # Return preprocessed image
 return img, denoised

def extract_barcode(image):
 # Decode all barcodes in the image
 barcodes = decode(image)
 # Initialize list to store barcode data
 barcode_data = []
 # Iterate through detected barcodes
 for barcode in barcodes:
 # Extract barcode data as string
 data = barcode.data.decode('utf-8')
 # Extract barcode type (CODE128, EAN13, etc.)
 barcode_type = barcode.type
 # Append to results
 barcode_data.append({'data': data, 'type': barcode_type})
 # Return list of barcode information
 return barcode_data

def detect_table_region(image):
 # Apply edge detection to find table boundaries
 edges = cv2.Canny(image, 50, 150, apertureSize=3)
 # Detect horizontal lines in the table
 horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
 # Detect vertical lines in the table
 vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
 # Extract horizontal lines
 horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
 # Extract vertical lines
 vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
 # Combine horizontal and vertical lines to get table structure
 table_mask = cv2.add(horizontal_lines, vertical_lines)
 # Return table mask
 return table_mask

def extract_table_data(image, preprocessed):
 # Configure Tesseract for table data extraction
 custom_config = r'--oem 3 --psm 6'
 # Extract text from the entire image
 text = pytesseract.image_to_string(preprocessed, config=custom_config)
 # Split text into lines
 lines = text.split('\n')
 # Initialize list to store table rows
 table_data = []
 # Flag to indicate if we're inside the table
 in_table = False
 # Iterate through each line
 for line in lines:
 # Skip empty lines
 if not line.strip():
 continue
 # Check if line contains table headers (Item, Batch, MFG, Exp, etc.)
 if 'Item' in line and 'Batch' in line:
 in_table = True
 continue
 # If we're in the table, parse the row
 if in_table:
 # Split by multiple spaces or tabs
 parts = re.split(r'\s{2,}|\t', line.strip())
 # If we have enough columns, it's a valid row
 if len(parts) >= 4:
 # Create row dictionary
 row = {
 'item': parts[0] if len(parts) > 0 else '',
 'batch': parts[1] if len(parts) > 1 else '',
 'mfg_date': parts[2] if len(parts) > 2 else '',
 'exp_date': parts[3] if len(parts) > 3 else '',
 'no_packs': parts[4] if len(parts) > 4 else ''
 }
 # Append to table data
 table_data.append(row)
 # Return extracted table data
 return table_data

def parse_dates(date_str):
 # Common date formats on labels
 date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d.%m.%Y']
 # Try each format
 for fmt in date_formats:
 try:
 # Attempt to parse the date
 parsed_date = datetime.strptime(date_str.strip(), fmt)
 # Return in ISO format
 return parsed_date.strftime('%Y-%m-%d')
 except ValueError:
 # Continue to next format if parsing fails
 continue
 # Return original string if no format matches
 return date_str

def extract_metadata(image, preprocessed):
 # Get image dimensions
 height, width = preprocessed.shape
 # Define ROI for metadata (right side of label)
 metadata_roi = preprocessed[0:height, int(width*0.6):width]
 # Extract text from metadata region
 metadata_text = pytesseract.image_to_string(metadata_roi)
 # Return metadata text
 return metadata_text

def process_pallet_label(image_path):
 # Preprocess the image
 original, preprocessed = preprocess_label_image(image_path)
 # Extract barcode information
 barcodes = extract_barcode(original)
 # Extract table data
 table_data = extract_table_data(original, preprocessed)
 # Parse dates in table data
 for row in table_data:
 row['mfg_date'] = parse_dates(row['mfg_date'])
 row['exp_date'] = parse_dates(row['exp_date'])
 # Extract metadata from right side
 metadata = extract_metadata(original, preprocessed)
 # Compile all extracted information
 result = {
 'barcodes': barcodes,
 'table_data': table_data,
 'metadata': metadata
 }
 # Return structured result
 return result

# Example usage
image_path = 'pallet_label.jpg'
# Process the pallet label
extracted_data = process_pallet_label(image_path)
# Print extracted barcode
print("Barcodes:", extracted_data['barcodes'])
# Print table data
print("\nTable Data:")
for row in extracted_data['table_data']:
 print(row)
# Print metadata
print("\nMetadata:", extracted_data['metadata'])
```

---

**💡 Explanation**:

**Algorithm Breakdown:**

**1. Preprocessing Pipeline:**
- **Grayscale conversion**: Reduces complexity, improves OCR accuracy
- **Adaptive thresholding**: Handles uneven lighting on the pallet
- **Denoising**: Removes camera noise and artifacts from cardboard texture
- **Time Complexity**: O(W×H) where W=width, H=height

**2. Barcode Extraction:**
- **pyzbar library**: Specialized for 1D/2D barcode detection
- Supports CODE128, EAN13, QR codes, etc.
- **Advantage**: More accurate than general OCR for barcodes
- **Time Complexity**: O(W×H) for detection

**3. Table Structure Detection:**
- **Morphological operations**: Detect horizontal/vertical lines
- **Line detection**: Identifies table grid structure
- **Cell extraction**: Isolate individual cells for OCR
- **Alternative**: Use deep learning models like TableNet or CascadeTabNet

**4. OCR with Tesseract:**
- **PSM 6**: Assumes uniform block of text (good for tables)
- **OEM 3**: Uses both legacy and LSTM engines
- **Custom config**: Optimizes for structured data
- **Accuracy**: ~85-95% depending on image quality

**5. Date Parsing:**
- **Multiple format support**: Handles DD-MM-YYYY, DD/MM/YYYY, etc.
- **Validation**: Ensures dates are valid
- **Normalization**: Converts to ISO format (YYYY-MM-DD)

**Expected Output Structure:**
```json
{
 "barcodes": [
 {"data": "123456789012", "type": "CODE128"}
 ],
 "table_data": [
 {
 "item": "220003256",
 "batch": "18000001",
 "mfg_date": "2024-01-15",
 "exp_date": "2024-07-15",
 "no_packs": "6"
 },
 {
 "item": "220003256",
 "batch": "24000002",
 "mfg_date": "2024-02-20",
 "exp_date": "2024-08-20",
 "no_packs": "24"
 }
 ],
 "metadata": "Pallet Label: 0.992\nPallet ID: 0.937\n..."
}
```

**Challenges & Solutions:**

| Challenge | Solution |
|-----------|----------|
| **Skewed image** | Apply perspective transform or deskewing |
| **Poor lighting** | Adaptive thresholding, CLAHE enhancement |
| **Barcode damage** | Try multiple barcode libraries (ZBar, OpenCV) |
| **Table misalignment** | Use deep learning table detection models |
| **OCR errors** | Post-processing with regex, spell-check |

**Production Improvements:**
- **Use EasyOCR or PaddleOCR** for better accuracy on complex fonts
- **Implement confidence scoring** to flag low-quality extractions
- **Add validation rules** (e.g., expiry date > manufacturing date)
- **Use LayoutLM or Donut** for end-to-end document understanding
- **Implement retry logic** with different preprocessing parameters

**Time Complexity**: **O(W×H×N)** where N = number of OCR passes
**Space Complexity**: **O(W×H)** for storing image data

This solution provides a robust pipeline for extracting structured data from pallet labels in a production warehouse environment, which aligns well with the **Vision Pipeline Engineer** role at KoiReader-Gyan!

---
