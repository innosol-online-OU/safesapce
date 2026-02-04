
import os

file_path = "src/core/protocols/latent_cloak.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Locate protect_phantom start
start_idx = -1
for i, line in enumerate(lines):
    if "def protect_phantom(self," in line:
        start_idx = i
        break

if start_idx == -1:
    print("Could not find protect_phantom definition.")
    exit(1)

# Locate the end of the block
# It ends before "        # Normalize 0.0 to 1.0"
# Which was around line 1241 in original file
end_idx = -1
for i in range(start_idx + 1, len(lines)):
    if "# Normalize 0.0 to 1.0" in line:

        end_idx = i
        break

if end_idx == -1:
    print("Could not find end of protect_phantom block.")
    exit(1)

print(f"Moving lines {start_idx} to {end_idx}...")

# Extract block
block = lines[start_idx:end_idx]
# Remove block from original location
remaining_lines = lines[:start_idx] + lines[end_idx:]

# Append block to the end
# Ensure we have some newlines
if not remaining_lines[-1].endswith("\n"):
    remaining_lines[-1] += "\n"

remaining_lines.append("\n\n")
remaining_lines.extend(block)

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(remaining_lines)

print("Done.")
