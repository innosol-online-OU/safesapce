
import os

file_path = "src/core/protocols/latent_cloak.py"

if not os.path.exists(file_path):
    print("File not found!")
    exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Read {len(lines)} lines.")

start_idx = -1
for i, line in enumerate(lines):
    if "def protect_phantom(self," in line:
        start_idx = i
        print(f"Found protect_phantom at line {i}: {line.strip()}")
        break

if start_idx == -1:
    print("Could not find protect_phantom definition.")
    exit(1)

end_idx = -1
for i in range(start_idx + 1, len(lines)):
    if "return final_pil" in line:
        end_idx = i + 1
        print(f"Found return final_pil at line {i}: {lines[i].strip()}")
        break
    # Debug: print every 50 lines to see where we are
    if (i - start_idx) % 50 == 0:
        print(f"Scanning line {i}: {lines[i].strip()[:20]}...")

if end_idx == -1:
    print("Could not find end of protect_phantom block.")
    exit(1)

print(f"Splitting at {end_idx}. Next line is: {lines[end_idx].strip()}")

block = lines[start_idx:end_idx]
remaining_lines = lines[:start_idx] + lines[end_idx:]

if not remaining_lines[-1].endswith("\n"):
    remaining_lines[-1] += "\n"

remaining_lines.append("\n\n")
remaining_lines.extend(block)

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(remaining_lines)

print("Done.")
