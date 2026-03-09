import os

def fast_count(path):
    if not os.path.exists(path):
        return "Directory Missing"
    
    print(f"Counting files in: {path}")
    count = 0
    stack = [path]

    # Iterative scan to avoid recursion depth limits
    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file() and entry.name.endswith(".json"):
                        count += 1
        except PermissionError:
            # Skip folders without permissions
            continue
        except FileNotFoundError:
            # Skip if folder disappears during scan
            continue

    return count

if __name__ == "__main__":
    p1 = "E:/Comic_Analysis_Results_v2/stage3_json"
    p2 = "E:/Comic_Analysis_Results_v2/detections"
    p3 = "E:/vlm_recycling_staging"
    
    count1 = fast_count(p1)
    print(f"Total Stage 3 JSONs: {count1}")
    
    count2 = fast_count(p2)
    print(f"Total Detection JSONs: {count2}")

    count3 = fast_count(p3)
    print(f"Total VLM JSONs: {count3}")
