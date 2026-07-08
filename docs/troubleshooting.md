# Troubleshooting

Common errors and how to fix them. Find your symptom below.

**Quick diagnosis:** if you're not sure where to start, run this:
```bash
conda activate bwsi-sgai
python -c "import numpy, torch, gymnasium, tkinter; print('All OK')"
```
The first import that fails is your problem.

---

## Setup errors

### The conda environment doesn't exist yet

**Symptom:** `conda activate bwsi-sgai` says `Could not find conda environment: bwsi-sgai`.

**Cause:** The environment was never created, or was created under a different name.

**Fix:**
```bash
# From the project root (where environment.yml lives):
conda env create -f environment.yml
conda activate bwsi-sgai
```

If the create command hangs or is very slow, install the faster solver first:
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```
Then retry `conda env create -f environment.yml`.

---

### `ModuleNotFoundError` — environment exists but packages are missing

**Symptom:** `conda activate bwsi-sgai` succeeds but `import torch` (or numpy, gymnasium, etc.) still fails.

**Cause:** The environment was created but the package install failed partway through. This can happen silently — conda marks the env as created even if pip dependencies failed.

**Fix:** Check what's actually installed, then install what's missing:
```bash
conda activate bwsi-sgai
conda list | grep torch       # check if torch is there
conda list | grep numpy
conda list | grep gymnasium
```

If a package is missing:
```bash
conda install numpy pandas matplotlib pillow
pip install torch gymnasium jupyterlab jupyterlab-rise
```

If the env is badly corrupted, delete and recreate:
```bash
conda deactivate
conda env remove -n bwsi-sgai
conda env create -f environment.yml
```

---

### `ModuleNotFoundError` — wrong Python interpreter in VS Code

**Symptom:** `conda activate bwsi-sgai` in the terminal works, but VS Code's Python extension still shows errors or uses the wrong Python.

**Cause:** VS Code caches the interpreter path. It may be pointing to the base env or system Python instead of `bwsi-sgai`.

**Fix:** `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the entry that includes `bwsi-sgai`. If it doesn't appear in the list, enter the path manually:
```
C:\Users\<your-username>\.conda\envs\bwsi-sgai\python.exe
```

Restart the VS Code terminal after switching (`Ctrl+Shift+~` or close and reopen the terminal panel).

---

### `ModuleNotFoundError: No module named 'gameplay'` (or any local module)

**Symptom:** Running a script directly (e.g., `python training/train_cnn.py`) raises `No module named 'gameplay'`, even though `gameplay/` exists in the project.

**Cause:** Python's import search starts from the directory where the script lives, not the project root. Running `python training/train_cnn.py` sets the working directory to `training/`, which can't see `gameplay/`.

**Fix:** Always run scripts from the project root:
```bash
cd /c/Users/<your-username>/projects/bwsi-sgai
python training/train_cnn.py
```

Or use `main.py` as the entry point (recommended):
```bash
python main.py train --type cnn
```

The training scripts also have a `sys.path` fix at the top that handles this automatically when run directly.

---

### `ImportError: DLL load failed while importing _multiarray_umath` (Windows — numpy)

**Symptom:** Importing numpy fails with a DLL error on Windows.

**Cause:** The conda environment's `Library\bin` folder is not in the Windows PATH. This is a one-time setup issue — the DLLs are installed but Windows can't find them.

**Fix:** Full instructions in `docs/windows-setup.md`. Short version:
1. `conda info --envs` — find the full path to the `bwsi-sgai` environment
2. Add `<env_path>\Library\bin` to User PATH in Windows environment variables
3. Fully restart VS Code (or restart the machine if the fix doesn't take)

Do not try to fix this by reinstalling numpy or recreating the environment. The PATH fix is the correct solution.

---

### `ImportError: DLL load failed` — PyTorch on Windows

**Symptom:** `import torch` fails with a DLL error, separate from the numpy issue above.

**Cause:** PyTorch on Windows requires the Microsoft Visual C++ Redistributable. A fresh or updated Windows install may not have it.

**Fix:**
1. Download and install **Microsoft Visual C++ Redistributable (latest)** from Microsoft's support site — search "Visual C++ Redistributable download"
2. Restart the machine
3. Test: `python -c "import torch; print(torch.__version__)"`

---

### `ModuleNotFoundError: No module named '_tkinter'`

**Symptom:** `import tkinter` fails even though tkinter is part of the Python standard library.

**Cause:** Tkinter depends on Tcl/Tk binaries that must be installed separately in conda. If the `tk` package is missing from the environment, tkinter can't load its C extension.

**Fix:**
```bash
conda activate bwsi-sgai
conda install tk
python -c "import tkinter; print('OK')"
```

If the game window opens but immediately crashes or shows blank, the Tcl/Tk version may be mismatched:
```bash
conda remove tk
conda install tk
```

---

### Anaconda itself is broken — reinstall

**Symptom:** Any of the following:
- `conda` command not found, even after a fresh terminal
- `CondaError` or `conda.exceptions` on every conda command
- Anaconda Navigator won't launch
- `conda activate` crashes the terminal
- `conda env create` always errors, even with a valid `environment.yml`

**Cause:** The Anaconda base installation is corrupted. This can happen from interrupted installs, Windows updates, or manual edits to the base environment. Unlike a broken project env (which you can delete and recreate), a broken base Anaconda can't be fixed with conda commands — conda is broken.

**Fix:** Uninstall and reinstall Anaconda from scratch.

1. Uninstall: **Settings → Apps → Anaconda** → Uninstall. On the uninstaller, check "Remove all user data."
2. Delete any leftover folders:
   - `C:\Users\<your-username>\anaconda3` or `C:\ProgramData\anaconda3`
   - `C:\Users\<your-username>\.conda`
3. Reinstall from [anaconda.com/download](https://www.anaconda.com/download). Choose "Just Me" (not All Users) to avoid permission issues.
4. Open a fresh Anaconda Prompt and verify: `conda --version`
5. Recreate the project environment: `conda env create -f environment.yml`

This takes 10–15 minutes. If you're in the middle of a session, let the instructor know before starting.

---

### `ModuleNotFoundError: No module named 'gym'`

**Symptom:** Code or a tutorial you found online uses `import gym`, but that fails.

**Cause:** OpenAI Gym (`gym`) is no longer maintained. This project uses its successor, `gymnasium` (Farama Foundation). They have the same API but different package names.

**Fix:** Replace `import gym` with `import gymnasium as gym` in any code you're adding. The project already uses `gymnasium` throughout.

---

---

## JupyterLab errors

### JupyterLab kernel won't start or immediately dies

**Symptom:** Opening a notebook shows "Kernel died, and restarted" or the kernel never connects.

**Cause:** JupyterLab is running but the `bwsi-sgai` environment is not registered as a kernel. JupyterLab needs a kernel spec installed for each conda env you want to use.

**Fix:**
```bash
conda activate bwsi-sgai
python -m ipykernel install --user --name bwsi-sgai --display-name "BWSI SGAI"
```

Then restart JupyterLab, open the notebook, and select **BWSI SGAI** from the kernel menu (top right of the notebook, or Kernel → Change Kernel).

Verify the kernel was registered:
```bash
jupyter kernelspec list
# Should show: bwsi-sgai    C:\Users\<user>\AppData\Roaming\jupyter\kernels\bwsi-sgai
```

---

### RISE slideshow button is missing in JupyterLab

**Symptom:** JupyterLab opens but there's no "Enter Slideshow" / bar chart button in the toolbar.

**Cause:** There are two separate RISE packages — `rise` (works only in classic Jupyter Notebook) and `jupyterlab-rise` (works in JupyterLab). If you have the wrong one installed, the button won't appear.

**Fix:**
```bash
conda activate bwsi-sgai
pip install jupyterlab-rise
jupyter lab build    # only needed if JupyterLab version < 3.x
```

Restart JupyterLab. The slideshow button (bar chart icon) should appear in the toolbar when a notebook is open.

---

### `ModuleNotFoundError` inside a Jupyter notebook

**Symptom:** `from gameplay import humanoid` fails inside a notebook, even though the same import works from the terminal.

**Cause:** Jupyter notebooks run with their working directory set to the notebook's folder, not the project root. If the notebook is in `lectures/`, Python can't see `gameplay/` which is in the project root.

**Fix:** Always launch JupyterLab from the project root:
```bash
cd /c/Users/<your-username>/projects/bwsi-sgai
conda activate bwsi-sgai
jupyter lab
```

If you still hit import errors inside a notebook, add this to the first cell:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))   # project root
```

---

---

## Runtime errors

### RL agent behaves erratically after changing `HEADCOUNT_WEIGHT`

**Cause:** The RL agent was trained under the old reward formula. Its policy is calibrated to the old weights — the mismatch produces incoherent behavior.

**Fix:** Retrain after any change to `HEADCOUNT_WEIGHT`:
```bash
python main.py train --type rl
# Or faster, for experimentation:
python main.py train --type rl -e 500
```

---

### Added a new action but clicking it does nothing

**Cause:** `ActionState` and `ActionCost` in `gameplay/enums.py` are separate enums. The action dispatcher (`map_do_action()`) is an if/elif chain — an unrecognized action falls through silently with no error.

**Fix:** Add matching entries to **both** enums:
```python
# In ActionState:
NEW_ACTION = "new_action"
# In ActionCost (same file, same name):
NEW_ACTION = 45   # minutes
```

---

### `datarow_to_value()` produces unexpected results

**Cause:** The function should return a float in `[0.0, 1.0]`. Values outside this range don't crash but bin to unexpected observation vector slots.

**Fix:** Clamp or compute your return value to stay within `[0.0, 1.0]`. The value is quantized into 5 levels — what matters is which bin it falls in, not the exact number.

---

### `RuntimeError: Error(s) in loading state_dict` — dimension mismatch

**Cause:** `OBS_SIZE` in `gameplay/constants.py` was changed after the RL weights were saved. The network input dimension changed, making the saved weights incompatible.

**Fix:** Retrain from scratch:
```bash
python main.py train --type rl
```
Old weights cannot be loaded after an obs size change.

---

### `FileNotFoundError: models/weights/baselineCNN.pth` (or `baselineRL.pth`)

**Cause:** Weight files are not included in the repository. You must train the models before running any mode that uses them.

| Mode | Needs CNN | Needs RL |
|------|-----------|----------|
| `human` | No | No |
| `human_cnn` | Yes | No |
| `human_rl` | Yes | Yes |
| `human_cnn_rl` | Yes | Yes |
| `machine` | Yes | Yes |
| `auto` | Yes | Yes |

**Fix:**
```bash
python main.py train --type cnn   # ~2 minutes
python main.py train --type rl    # ~20–40 minutes; use -e 500 for a faster baseline
```

---

---

## Data and logging

### `log.csv` contains data from mixed experiments

**Cause:** `log.csv` and `summary.csv` append across every run — they never reset automatically.

**Fix:** Delete both files before starting a new experiment condition:
```bash
rm log.csv summary.csv
```
Then run with `--log --seed <N>` to keep results reproducible.

---

### Notebook and script show different hyperparameter values

**Cause:** `training/train_rl.ipynb` and `training/train_rl.py` have separate copies of the hyperparameter block. Editing one doesn't update the other.

**Fix:** Pick one interface and make all changes there. If you've edited both, compare the hyperparameter blocks manually. The canonical defaults are in the `.py` file.

---

---

## Diagnosing unexpected game behavior

### Is the problem in the CNN or the RL agent?

Swap `CNNClassifier` for `OracleClassifier` — a perfect classifier that reads ground-truth labels directly, with no weights required. If the game plays well with oracle but poorly with CNN, the CNN is the bottleneck. If it plays poorly with both, the RL policy needs work.

Edit `main.py` to swap in the oracle:
```python
# Add near the top of main.py:
from models.OracleClassifier import OracleClassifier

# In _make_components() and _run_auto(), replace:
CNNClassifier(img_data_root=data_dir)
# With:
OracleClassifier()
```

`OracleClassifier` takes no arguments and never raises `FileNotFoundError`. Revert both edits before running real experiments.
