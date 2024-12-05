#!/usr/bin/env python3
import sys
import json

def reducer():
    combined_data = []
    
    for line in sys.stdin:
        try:
            # Parse each line as JSON 
            record = json.loads(line.strip())
            combined_data.append(record)
        except json.JSONDecodeError:
            continue  # Skip invalid lines

    # Output a single JSON array
    print(json.dumps(combined_data))

if __name__ == "__main__":
    reducer()
