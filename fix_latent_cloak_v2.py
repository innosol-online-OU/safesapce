
import os

file_path = "src/core/protocols/latent_cloak.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if "def protect_phantom(self," in line:
        start_idx = i
        break

if start_idx == -1:
    print("Could not find protect_phantom definition.")
    exit(1)

# Locate the return statement of protect_phantom
end_idx = -1
for i in range(start_idx + 1, len(lines)):
    # We look for "return final_pil" with indentation
    if "return final_pil" in lines[i]:

        end_idx = i + 1 # Include this line
        break

if end_idx == -1:
    print("Could not find end of protect_phantom block (return final_pil).")
    exit(1)

print(f"Moving lines {start_idx} to {end_idx}...")
# Verify the lines around cut
print(f"Last line of block: {lines[end_idx-1]}")
print(f"First line of remaining (should be 'Normalize...'): {lines[end_idx]}")

block = lines[start_idx:end_idx]
remaining_lines = lines[:start_idx] + lines[end_idx:]

if not remaining_lines[-1].endswith("\n"):
    remaining_lines[-1] += "\n"

remaining_lines.append("\n\n")
remaining_lines.extend(block)

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(remaining_lines)

print("Done.")
