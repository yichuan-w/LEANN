# Fix for Permission Error in Interactive Mode

## Problem
When running `leann ask --interactive`, you get:
```
PermissionError: [Errno 1] Operation not permitted
```

## Solution Applied
Created the history file with proper permissions:
```bash
touch ~/.leann_history
chmod 644 ~/.leann_history
```

## Alternative: Use Non-Interactive Mode

If interactive mode still has issues, use non-interactive mode instead:

**Instead of:**
```bash
leann ask my-docs --interactive --model gemma3:4b
```

**Use:**
```bash
leann ask my-docs "your question here" --model gemma3:4b
```

You can ask multiple questions by running the command multiple times, or create a simple script:

```bash
#!/bin/bash
cd ~/LEANN
source .venv/bin/activate

leann ask my-docs "What did I write about project X?" --model gemma3:4b
leann ask my-docs "What are the main topics?" --model gemma3:4b
leann ask my-docs "Summarize the documents" --model gemma3:4b
```

## If Permission Error Persists

Try these steps in your terminal:

1. **Check file permissions:**
   ```bash
   ls -la ~/.leann_history
   ```

2. **Fix permissions if needed:**
   ```bash
   chmod 644 ~/.leann_history
   ```

3. **Or remove and recreate:**
   ```bash
   rm ~/.leann_history
   touch ~/.leann_history
   chmod 644 ~/.leann_history
   ```

4. **Check if it's a macOS security issue:**
   - Go to System Settings > Privacy & Security > Files and Folders
   - Make sure Terminal (or your terminal app) has access to your home directory

## Quick Test

Test if it works now:
```bash
cd ~/LEANN
source .venv/bin/activate
leann ask my-docs "test question" --model gemma3:4b
```

If that works, try interactive mode:
```bash
leann ask my-docs --interactive --model gemma3:4b
```
