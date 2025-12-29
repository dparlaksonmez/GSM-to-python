# Installation Guide (Windows + WSL)

This project runs best in Linux. On Windows, the easiest way is **WSL2** (Windows Subsystem for Linux).

If you are new to this:
- **Windows** is the “outside building”.
- **WSL (Ubuntu)** is a “small Linux lab inside Windows”.
- We will run the pipeline inside that Linux lab.

---

## 0) What you need

- Windows 10/11
- Internet connection
- Enough disk space for Python packages and datasets

---

## 1) Install WSL + Ubuntu

### Option A (recommended): one command

1. Open **PowerShell as Administrator**
2. Run:

```powershell
wsl --install
```

3. Restart your computer when it asks.

### Option B (if Option A does not work)

Search Microsoft docs for “Install WSL” and follow the official steps.

---

## 2) Open Ubuntu (WSL)

- Open **Start Menu** → search **Ubuntu** → open it.
- The first time, it will ask you to create a Linux username + password.
  - The password will not show when typing (this is normal).

---

## 3) Install basic tools inside Ubuntu

In the Ubuntu terminal:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

(If `sudo` asks for your password, use the password you created for Ubuntu.)

---

## 4) Install VS Code + WSL extension

### Install VS Code

- Download and install Visual Studio Code from the official Microsoft website.

### Install the WSL extension (important)

In VS Code:
1. Open Extensions (left sidebar)
2. Search: **WSL**
3. Install: **Remote - WSL**

Why? Because it lets VS Code edit and run code *inside* Ubuntu/WSL.

---

## 5) Get the project into WSL (VS Code way)

If you are new, this is the easiest method because VS Code guides you.

### Step 1: Open a WSL window

In VS Code:
1. Press `Ctrl+Shift+P` (Command Palette)
2. Run: **WSL: New WSL Window**

You should now see `WSL: Ubuntu` in the bottom-left.

### Step 2: Clone using the VS Code interface

In that WSL window:
1. Press `Ctrl+Shift+P`
2. Run: **Git: Clone**
3. Paste the repo URL: `https://github.com/shiny-apricot/GSM-to-python.git`
4. Choose a folder inside Linux home, like `/home/<you>/` (VS Code will show it as `~`)
5. When prompted, click **Open** to open the cloned repo

Screenshot placeholder (optional):
- `![VS Code: Git Clone](images/vscode-git-clone.png)`

---

## 6) (Fallback) Clone using Ubuntu terminal

Inside **Ubuntu terminal** (recommended location is your Linux home directory):

```bash
cd ~
git clone https://github.com/shiny-apricot/GSM-to-python.git
cd GSM-to-python
```

Tip: Avoid cloning into `/mnt/c/...` at first. For beginners, keeping the code inside Linux (`/home/<you>/...`) is simpler and usually faster.

---

## 7) Create a Python virtual environment (VS Code way)

A virtual environment is like a **clean bench space** in the lab: it keeps packages for this project separate from other projects.

In VS Code:
1. Press `Ctrl+Shift+P`
2. Run: **Python: Create Environment**
3. Choose **Venv**
4. Select the default Python interpreter it suggests
5. Wait for it to finish

Then in VS Code, check the bottom-right Python version/venv indicator.

Screenshot placeholder (optional):
- `![VS Code: Python Create Environment](images/vscode-python-create-env.png)`

---

## 8) (Fallback) Create a virtual environment in terminal

From the project root (Ubuntu terminal or VS Code terminal):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt.

---

## 9) Install Python dependencies (VS Code terminal)

Installing packages still runs a command, but you can do it from VS Code without “learning terminal navigation”.

In VS Code:
1. **Terminal → New Terminal**
2. Make sure you are in the project folder (VS Code usually opens the terminal there)
3. If needed, activate the environment (VS Code may do this automatically)
4. Run:

```bash
pip install -r dependencies.txt
```

If installation fails, check: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 10) Open the project in VS Code (inside WSL)

If you used the VS Code clone flow, you are already done.

If you cloned in a terminal, you can open VS Code from Ubuntu.

From the project root (terminal), run:

```bash
code .
```

VS Code should show you are connected to **WSL: Ubuntu** (bottom-left).

---

## 11) Quick “it works” check

### Run the UI (Streamlit)

```bash
streamlit run src/ui/app.py
```

It should print a local URL like `http://localhost:8501`.
Open that in your browser.

---

## 12) Updating the project later (VS Code way)

In VS Code:
1. Open **Source Control** (left sidebar icon)
2. Click **…** (More Actions)
3. Choose **Pull**

Then, if dependencies changed, run the install command again in the VS Code terminal.

---

## 13) (Fallback) Updating in terminal

Whenever you come back to the project:

```bash
cd ~/GSM-to-python
source .venv/bin/activate

git pull
pip install -r dependencies.txt
```

---

## Screenshot placeholders (optional)

If you want to add screenshots later:

- `![Ubuntu terminal opened](images/ubuntu-open.png)`
- `![VS Code connected to WSL](images/vscode-wsl.png)`
