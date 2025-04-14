def merge_minus_to_equals(symbols):
    """
    Merge pairs of vertically stacked '-' symbols into a single '=' symbol.
    Args:
        symbols: List of dicts with 'bbox': [x_min, y_min, x_max, y_max], 'type': str
    Returns:
        Updated list of symbols with merged '=' symbols.
    """
    if not symbols:
        return symbols
    
    # Calculate average symbol height and width for thresholds
    avg_height = sum(s['bbox'][3] - s['bbox'][1] for s in symbols) / max(len(symbols), 1)
    avg_width = sum(s['bbox'][2] - s['bbox'][0] for s in symbols) / max(len(symbols), 1)
    
    new_symbols = []
    used_indices = set()
    
    for i, s1 in enumerate(symbols):
        if i in used_indices or s1['type'] != '.':
            continue
        for j, s2 in enumerate(symbols):
            if j <= i or j in used_indices or s2['type'] != '.':
                continue
            # Check if symbols are vertically stacked and horizontally aligned
            x_min1, y_min1, x_max1, y_max1 = s1['bbox']
            x_min2, y_min2, x_max2, y_max2 = s2['bbox']
            if (abs(x_min1 - x_min2) < avg_width * 0.5 and
                abs(y_min1 - y_min2) < avg_height * 1.5 and
                y_min1 != y_min2):  # Ensure they’re not at the same y
                # Merge into '='
                new_bbox = [
                    min(x_min1, x_min2),
                    min(y_min1, y_min2),
                    max(x_max1, x_max2),
                    max(y_max1, y_max2)
                ]
                new_symbols.append({'bbox': new_bbox, 'type': '='})
                used_indices.add(i)
                used_indices.add(j)
                break
    
    # Add unmerged symbols
    for i, s in enumerate(symbols):
        if i not in used_indices:
            new_symbols.append(s)
    
    return new_symbols

def symbols_to_string(symbols):
    """
    Convert a list of symbols with bounding boxes and types into a string.
    Args:
        symbols: List of dicts, each with 'bbox': [x_min, y_min, x_max, y_max], 'type': str
    Returns:
        String representation of symbols, grouped by rows.
    """

    symbols = merge_minus_to_equals(symbols)

    if not symbols:
        return ""
    
    # Calculate average symbol height for row grouping threshold
    avg_height = sum(s['bbox'][3] - s['bbox'][1] for s in symbols) / max(len(symbols), 1)
    
    # Group symbols into rows based on y_min/y_max overlap
    rows = []
    sorted_symbols = sorted(symbols, key=lambda s: s['bbox'][1])  # Sort by y_min initially
    current_row = [sorted_symbols[0]]
    current_y_min = sorted_symbols[0]['bbox'][1]
    
    for symbol in sorted_symbols[1:]:
        y_min = symbol['bbox'][1]
        # If y_min is close to current row (within avg_height), add to current row
        if y_min < current_y_min + avg_height:
            current_row.append(symbol)
        else:
            # Sort current row by x_min and store
            rows.append(sorted(current_row, key=lambda s: s['bbox'][0]))
            current_row = [symbol]
            current_y_min = y_min
    
    # Add the last row
    if current_row:
        rows.append(sorted(current_row, key=lambda s: s['bbox'][0]))
    
    # Concatenate symbols row by row
    result = ""
    for row in rows:
        row_string = "".join(s['type'] for s in row)
        result += row_string + "\n"
    
    return result.strip()

# Example usage
def main():
    # Sample input: list of symbols with bounding boxes and types
    symbols = [
        {'bbox': [10, 10, 20, 20], 'type': 'H'},
        {'bbox': [30,11, 40, 20], 'type': 'i'},
        {'bbox': [50, 10, 60, 20], 'type': '!'},
        {'bbox': [10, 30, 20, 40], 'type': 'B'},
        {'bbox': [30, 30, 40, 40], 'type': 'y'},
        {'bbox': [50, 30, 60, 40], 'type': 'e'}
    ]
    
    # Convert symbols to string
    output = symbols_to_string(symbols)
    print("Input symbols:")
    for s in symbols:
        print(f"Type: {s['type']}, BBox: {s['bbox']}")
    print("\nOutput string:")
    print(output)

if __name__ == "__main__":
    main()