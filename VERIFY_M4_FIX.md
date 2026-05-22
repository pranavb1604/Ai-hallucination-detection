# Verify M4 Fix is Working

## Quick Test Command

Run this in your terminal (with venv activated):

```bash
.\.venv\Scripts\Activate.ps1
python debug_m4.py
```

## Expected Output

You should see:
```
✓ M4 Score: 0.9822
✓ Verdict: Strongly entailed by evidence
✓ Claim 1: The biggest planet in our solar system is Jupiter.
   Entailment: 0.9822
```

## If You See This

✅ **M4 is working correctly!**

The issue is that Streamlit is caching the old module.

## Force Streamlit to Use New Code

### Option 1: Clear Streamlit Cache (IN THE BROWSER)
1. In the Streamlit app, click the **☰ menu** (top right)
2. Click **"Clear cache"**
3. Click **"Clear cache"** again to confirm
4. Click **"Rerun"**

### Option 2: Hard Refresh Browser
1. Press **Ctrl + Shift + R** (Windows/Linux)
2. Or **Cmd + Shift + R** (Mac)
3. This forces browser to reload everything

### Option 3: Use Incognito/Private Window
1. Open a **new incognito/private browser window**
2. Go to http://localhost:8501
3. Try the question again

### Option 4: Restart Everything
```bash
# Stop Streamlit (Ctrl+C in the terminal running it)
# Then clear Python cache
Remove-Item -Path "modules\__pycache__\*" -Force
# Restart Streamlit
.\.venv\Scripts\Activate.ps1
streamlit run app\streamlit_app.py
```

## What Was Fixed

1. **Label case**: Changed from `'ENTAILMENT'` to `'entailment'`
2. **Evidence truncation**: Now uses only first 3 sentences (~400 chars)
3. **Query improvement**: Better Wikipedia page selection

## Files Modified

- `modules/m4_entailment.py` - Fixed NLI label matching and truncation
- `modules/m2_grounding.py` - Improved query generation
- `app/streamlit_app.py` - Added module reloading

## Still Showing 0?

If after ALL of the above you still see 0, run:

```bash
.\.venv\Scripts\Activate.ps1
python -c "from modules.m4_entailment import _entailment_prob; print('Score:', _entailment_prob('Jupiter is the largest planet in the Solar System', 'Jupiter is big'))"
```

This should print: `Score: 0.98...`

If this works but Streamlit still shows 0, the issue is definitely Streamlit caching.
