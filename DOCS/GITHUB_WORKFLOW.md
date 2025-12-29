# Git + GitHub Workflow (for Lab Teams)

GitHub is where we store and share the project.
Git is the tool that records changes.

Analogy:
- **Project files** = your experiment materials
- **Git commit** = a lab notebook entry (“what I changed and why”)
- **Branch** = a separate bench where you can work without disturbing others
- **Pull Request (PR)** = asking the team to review your work before it becomes the official protocol

---

## 1) One-time setup

### Recommended: let VS Code guide you

When you commit for the first time, VS Code may ask for your name/email.
Fill it in when prompted.

If VS Code does not ask and commits fail due to missing identity, use the terminal fallback below.

### (Fallback) Configure name/email (terminal)

Open a VS Code terminal (**Terminal → New Terminal**) and run:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Authenticate with GitHub

Simplest for beginners: use HTTPS (no SSH keys needed).
VS Code will usually help you sign in when you push.

---

## 2) Get the project on your computer

### Recommended: Clone with VS Code (no terminal navigation)

In VS Code (in a WSL window if you are on Windows):
1. Press `Ctrl+Shift+P`
2. Run: **Git: Clone**
3. Paste the repository URL
4. Choose a folder
5. Click **Open** when VS Code asks

Screenshot placeholder (optional):
- `![VS Code: Clone repository](images/vscode-clone.png)`

### (Fallback) Clone using terminal

In terminal:

```bash
cd ~
git clone https://github.com/shiny-apricot/GSM-to-python.git
cd GSM-to-python
```

### If you want to contribute (recommended)

You have two common cases:

#### Case A: You are a collaborator (you can push branches)
You can use the repo directly.

#### Case B: You are not a collaborator
Use a **fork** (your own copy on GitHub):
1. On GitHub, click **Fork**
2. Clone *your fork* URL instead of the main repo URL

Screenshot placeholder (optional):
- `![Fork button location](images/github-fork.png)`

---

## 3) Daily workflow (the safe way)

This section is written “VS Code first”. Terminal commands are included as a backup.

### Step 1: Pull latest changes (VS Code)

1. Click **Source Control** (left sidebar)
2. Click **…** (More Actions)
3. Choose **Pull**

If you have multiple branches, make sure you are on `main` first.

#### (Fallback) Terminal pull

```bash
git checkout main
git pull
```

### Step 2: Create a new branch for your task (VS Code)

Name it like a short sentence:

1. Look at the bottom-left status bar: you’ll see the current branch name
2. Click the branch name
3. Choose **Create new branch…**
4. Type a name like `fix-ui-upload`

#### (Fallback) Terminal branch

```bash
git checkout -b fix-ui-upload
```

### Step 3: Make your changes

Edit files in VS Code.

### Step 4: See what changed (VS Code)

1. Open **Source Control**
2. You will see a list of changed files
3. Click a file to see the diff (before/after)

This is like reviewing what changed in your protocol before signing it.

#### (Fallback) Terminal status

```bash
git status
```

### Step 5: Save the changes as a commit (VS Code)

1. Open **Source Control**
2. Hover a file and click **+** (Stage) to select it
	- Or click **Stage All** if everything is part of the same change
3. Write a message in the message box (example: `Fix upload validation in UI`)
4. Click **Commit**

#### (Fallback) Terminal commit

```bash
git add -A
git commit -m "Fix upload validation in UI"
```

Commit message tip:
- Good: “Fix UI upload validation”
- Avoid: “update” / “stuff”

### Step 6: Push your branch to GitHub (VS Code)

1. Open **Source Control**
2. Click **Sync Changes** or **Push** (wording depends on VS Code version)
3. If asked to sign in to GitHub, follow the prompts

#### (Fallback) Terminal push

```bash
git push -u origin fix-ui-upload
```

### Step 7: Open a Pull Request (PR)

#### Option A (simple): GitHub website

After pushing, GitHub will usually show a button like “Compare & pull request”.

#### Option B (inside VS Code): GitHub Pull Requests extension

If your team uses it:
1. Install the extension **GitHub Pull Requests**
2. Sign in to GitHub in VS Code
3. Open the Pull Requests view and create a PR from your branch

In your PR description, write:
- What you changed
- Why you changed it
- How someone can test it

Screenshot placeholder (optional):
- `![Open PR button](images/github-open-pr.png)`

---

## 4) If your branch gets behind main

git checkout fix-ui-upload
git merge main
### VS Code way

1. Checkout `main` (click branch name bottom-left)
2. Pull (Source Control → **…** → Pull)
3. Checkout your feature branch again
4. Merge `main` into your branch:
	- Source Control → **…** → Branch → Merge Branch… → select `main`

### (Fallback) Terminal way

```bash
git checkout main
git pull
git checkout fix-ui-upload
git merge main
```

If merge conflicts happen, ask for help. Conflicts are not “dangerous”; they are Git asking you to choose between two edits.

---

## 5) Very small emergency undo

### Discard local edits

VS Code way:
1. Source Control
2. Right-click a changed file → **Discard Changes**

Terminal fallback:

```bash
git restore .
```

If you committed but want to undo the last commit (local only):

```bash
git reset --soft HEAD~1
```

Use these carefully.
