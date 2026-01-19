# Fix Interactive Mode Permission Error

## The Problem
macOS is blocking readline from accessing `~/.leann_history` even with correct permissions. This is a macOS security feature.

## Solution 1: Fix Permissions (Try This First)

Run in your terminal:
```bash
chmod 666 ~/.leann_history
```

Or remove and recreate:
```bash
rm ~/.leann_history
touch ~/.leann_history
chmod 666 ~/.leann_history
```

## Solution 2: Use Non-Interactive Scripts (Recommended)

Instead of `leann ask --interactive`, use these scripts:

### Ask a Single Question:
```bash
python ask_non_interactive.py "What did I write about project X?"
```

### Search:
```bash
python quick_search.py "project X"
```

### Ask Multiple Questions (Create a script):
```bash
#!/bin/bash
cd ~/LEANN
source .venv/bin/activate

python ask_non_interactive.py "What did I write about project X?"
python ask_non_interactive.py "What are the main topics?"
python ask_non_interactive.py "Summarize the documents"
```

## Solution 3: Check macOS Privacy Settings

1. Go to **System Settings** > **Privacy & Security** > **Files and Folders**
2. Make sure your terminal app has access to:
   - ✅ Home Directory
   - ✅ Documents Folder

3. If using Terminal.app, you might need to grant Full Disk Access:
   - System Settings > Privacy & Security > Full Disk Access
   - Add Terminal.app

## Solution 4: Use a Different Terminal

Try using a different terminal app (iTerm2, Warp, etc.) which might have different permission settings.

## Quick Test

After fixing permissions, test:
```bash
cd ~/LEANN
source .venv/bin/activate
leann ask my-docs "test" --model gemma3:4b
```

If that works (non-interactive), then try:
```bash
leann ask my-docs --interactive --model gemma3:4b
```

## Alternative: Patch LEANN (Advanced)

If nothing else works, you can modify the interactive_utils.py to catch the permission error:

```python
# In ~/LEANN/.venv/lib/python3.11/site-packages/leann/interactive_utils.py
# Around line 69, change:
try:
    readline.read_history_file(str(history_file))
    readline.set_history_length(1000)
except (FileNotFoundError, PermissionError):  # Add PermissionError
    pass
```

But this requires modifying the installed package, which isn't ideal.

## Recommended Approach

**Just use the non-interactive scripts** - they work perfectly and avoid all permission issues:

```bash
# Search
python quick_search.py "your query"

# Ask
python ask_non_interactive.py "your question"
```

These scripts are faster, more reliable, and don't have permission issues!
