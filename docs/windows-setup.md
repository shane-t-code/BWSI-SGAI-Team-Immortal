# Windows Setup — One-Time Steps

These steps are required once per machine on Windows. They fix a numpy DLL
crash that occurs when VSCode launches Python without the right libraries in
your PATH.

---

## Prerequisites

- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git installed and the repo cloned

---

## Step 1 — Create the conda environment

Open **Anaconda Prompt** (not regular Command Prompt) and navigate to the project folder:

```bat
cd path\to\bwsi-sgai
conda env create -f environment.yml
```

Replace `path\to\bwsi-sgai` with the actual path to your cloned repo.
This installs all dependencies including PyTorch. It may take a few minutes.

---

## Step 2 — Fix numpy DLL loading on Windows

numpy needs a specific folder in your PATH to find its math libraries
(OpenBLAS). Without it you get a `DLL load failed` crash whenever Python
imports numpy — including every time you run the game or tests in VSCode.

1. Open **Anaconda Prompt** — search the Start menu for it. Do NOT use
   PowerShell or regular Command Prompt.

2. Find your environment path:

   ```bat
   conda info --envs
   ```

   Look for the `bwsi-sgai` line. Copy the full path shown next to it.
   It will look something like `C:\Users\yourname\.conda\envs\bwsi-sgai`
   but the exact location may differ on your machine.

3. Append `\Library\bin` to that path (no trailing backslash). This subfolder
   holds the binary libraries numpy needs. For example:

   ```
   C:\Users\yourname\.conda\envs\bwsi-sgai\Library\bin
   ```

4. Press **Win+S** and search for **"Edit environment variables for your
   account"**. Open it.

5. In the top section labeled **User variables** (not System variables), click
   **Path** → **Edit** → **New**.

6. Paste the path from step 3 → click **OK** on all open dialogs.

7. Restart VSCode completely. PATH changes don't apply to terminals that are
   already open. If you still get a DLL error after restarting VSCode, restart
   your computer — some Windows configurations require a full logout/login for
   PATH changes to take effect.

8. In a VSCode terminal, verify numpy loads:

   ```bat
   python -c "import numpy; print('OK')"
   ```

   If it prints `OK`, you're done.

---

## Step 3 — Set VSCode's Python interpreter

1. Open the project folder in VSCode
2. Press `Ctrl+Shift+P` → type **Python: Select Interpreter**
3. Choose the `bwsi-sgai` conda environment from the list

If `bwsi-sgai` doesn't appear in the list, press `Ctrl+Shift+P` →
**Python: Clear Cache and Reload**, or close and reopen VSCode, then try again.

VSCode will remember this choice for the project.

---

## Step 4 — Verify everything works

In VSCode, open the **Run and Debug** panel (`Ctrl+Shift+D`) and run
**"pytest: run tests"**. All tests should pass (no failures or errors).

Or from **Anaconda Prompt**:

```bat
conda activate bwsi-sgai
cd path\to\bwsi-sgai
python -m pytest tests\
```

Replace `path\to\bwsi-sgai` with the actual path to your cloned repo.

---

## Troubleshooting

**numpy fails to import (`DLL load failed` or `ModuleNotFoundError: numpy`)**
The PATH entry is missing or incorrect. Work through this checklist:
1. Close and fully restart VSCode — PATH changes don't apply to open terminals.
2. Run `conda info --envs` in **Anaconda Prompt** to get the exact path for
   `bwsi-sgai`. Don't guess or copy from an example.
3. Confirm the path ends with `\Library\bin` and has no trailing backslash.
4. Confirm you added it under **User variables**, not System variables.
5. Repeat Step 2 from scratch if anything looks off.
6. If the error persists after all of the above, restart your computer (not
   just VSCode) — some Windows configurations require a full logout/login for
   PATH changes to take effect.

**`%CONDA_PREFIX%` printed literally in your terminal**
This happens if you try to use `%CONDA_PREFIX%\Library\bin` as a shortcut
path in PowerShell or regular CMD. Those shells don't expand conda's variables.
Use the literal path from `conda info --envs`, and always use **Anaconda Prompt**.

**`ModuleNotFoundError` on any import (not numpy)**
Wrong Python interpreter selected in VSCode. Repeat Step 3.

**`bwsi-sgai` doesn't appear in the VSCode interpreter list**
Press `Ctrl+Shift+P` → **Python: Clear Cache and Reload**, then try Step 3
again. If it still doesn't appear, confirm the environment was created
successfully by running `conda info --envs` in Anaconda Prompt.

**Conda command not found**
Use **Anaconda Prompt** (search Start menu), not regular Command Prompt or
PowerShell.
