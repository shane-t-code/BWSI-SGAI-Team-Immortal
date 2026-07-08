# Getting Set Up

By the end of this page you'll have the game running on your machine.

---

## What you need

Install these before you start:

- **VS Code** — [code.visualstudio.com](https://code.visualstudio.com/)
- **Anaconda** (includes conda and Python) — [anaconda.com/download](https://www.anaconda.com/download)
- **Git** — to clone your team's repository. Pick one:
  - Command line (all platforms): [git-scm.com/downloads](https://git-scm.com/downloads)
  - Windows GUI: [GitHub Desktop](https://desktop.github.com/) or [GitKraken](https://www.gitkraken.com/)
  - Mac GUI: [GitHub Desktop](https://desktop.github.com/) or install via Homebrew: `brew install git`

There are other tools that work (venv instead of conda, PyCharm instead of VS Code), but this guide only covers these.

### Verify your installs

Before going further, confirm each tool is installed. Open Anaconda Prompt and run:

```
git --version
conda --version
code --version
```

Each should print a version number. If any command says "not found" or "not recognized," revisit the install for that tool before continuing.

---

## Step-by-step

### 1. Open Anaconda Prompt

Search the Start menu for **Anaconda Prompt** and open it. Use this — not PowerShell, not Command Prompt — for the steps below.

### 2. Clone your team's repository

Your team has its own fork of the project on MIT's GitHub. Your instructor will give you the link on the first day. It will look like one of these:

```
# SSH (recommended if you have an SSH key set up):
git@github.mit.edu:BWSI-SGAI-2026/team-name.git

# HTTPS (works without an SSH key):
https://github.mit.edu/BWSI-SGAI-2026/team-name.git
```

Clone it and move into the folder:

```
git clone <your-team-repo-url>
cd team-name
```

### 3. Create the project Python environment

The project includes a file called `environment.yml` that lists every Python package the game needs. This command reads it and sets up an isolated Python installation (called an **environment**) just for this project:

```
conda env create -f environment.yml
```

This takes a few minutes. You'll see a lot of text scrolling by — that's normal. Wait for it to finish before moving on.

### 4. Activate the environment

```
conda activate bwsi-sgai
```

Your prompt should now show `(bwsi-sgai)` on the left. Run this every time you open a new Anaconda Prompt.

### 5. Fix the Windows numpy issue (one time only)

NumPy needs a one-time fix on Windows. Follow the steps in [docs/windows-setup.md](windows-setup.md). Come back here when you're done.

### 6. Open the project in VS Code

```
code .
```

This opens VS Code in the project folder. (The `.` means "this folder.")

### 7. Run the game

Open the **Run and Debug** panel (`Ctrl+Shift+D`). You'll see a dropdown list of modes at the top. Pick **"human: pure human play"** and press the green play button.

---

## Quick check

If the game window opens — you're set.

If something went wrong, check the terminal at the bottom of VS Code for an error message and ask your instructor.

---

## Opening lecture notebooks

The notebooks for each module are in the `lectures/` folder. With the environment active in Anaconda Prompt:

```
jupyter lab
```

JupyterLab opens in your browser. Navigate to `lectures/` in the file panel on the left.

---

## Running the tests

```
pytest
```

All tests should pass. If any fail, let your instructor know.

---

## Troubleshooting

**Prompt still shows `(base)` instead of `(bwsi-sgai)`**

You forgot to activate the environment. Run:

```
conda activate bwsi-sgai
```

Do this every time you open a new Anaconda Prompt.

---

**`ModuleNotFoundError` when running the game**

Either the environment isn't active (see above), or VS Code is using the wrong Python interpreter.

Fix in VS Code: open the Command Palette (`Ctrl+Shift+P`), search **Python: Select Interpreter**, and pick the one that says `bwsi-sgai`.

---

**NumPy crashes on Windows with a DLL error**

Follow the steps in [docs/windows-setup.md](windows-setup.md) — this is a one-time fix for Windows machines.

---

**`FileNotFoundError` for model weights**

The pre-trained model files aren't included in the repo. Train them first:

```
python main.py train --type cnn
python main.py train --type rl
```

---

**`code .` is not recognized**

VS Code isn't on your PATH. Fix it from inside VS Code: open the Command Palette (`Ctrl+Shift+P`), search **Shell Command: Install 'code' command in PATH**, and run it. Then restart Anaconda Prompt.

---

**Something else is wrong**

Read the error message in the terminal carefully — it usually says exactly what file or package is missing. If you're stuck, ask your instructor.
