import os

model_path = "lstm_[1468447442.91]_39_80_0.921.p"

# Read the corrupted binary data
with open(model_path, 'rb') as f:
    content = f.read()

# Replace Windows line endings (\r\n) with Unix line endings (\n)
fixed_content = content.replace(b'\r\n', b'\n')

# Overwrite the file with the fixed data
with open(model_path, 'wb') as f:
    f.write(fixed_content)

print("File fixed successfully!")