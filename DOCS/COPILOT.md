# GitHub Copilot Pro Guide (Simple + Practical)

GitHub Copilot is an AI assistant for coding.

Analogy: it’s like **autocorrect + a helpful lab colleague** who can draft code, suggest next lines, and explain errors.

Important reality check:
- Copilot can be very helpful, but it can also be wrong.
- Treat its suggestions like a *draft protocol* that you still must review.

---

## 1) What you need

- A GitHub account
- GitHub Copilot Pro subscription (or access via your organization if your lab provides it)
- VS Code installed

---

## 2) Subscribe to Copilot Pro

1. Log into GitHub
2. Go to GitHub Copilot settings / billing
3. Subscribe to **Copilot Pro**

(Exact pages can change over time; the easiest is to search “GitHub Copilot Pro subscribe”.)

Screenshot placeholder (optional):
- `![Copilot subscription page](images/copilot-subscribe.png)`

---

## 3) Install Copilot in VS Code

In VS Code:
1. Open Extensions
2. Search: **GitHub Copilot**
3. Install it
4. Also install: **GitHub Copilot Chat** (if it is not included)

Then:
- Sign in to GitHub when VS Code asks

Tip: You can confirm you’re signed in by clicking the Accounts icon (bottom-left) in VS Code.

Screenshot placeholder (optional):
- `![VS Code Copilot extension install](images/vscode-copilot-install.png)`

---

## 4) How to use Copilot (3 common ways)

### A) Inline suggestions while you type

- Start writing a function
- Copilot will suggest the rest
- Press **Tab** to accept (or keep typing to ignore)

### B) Copilot Chat: ask questions about code

Where to find it:
- In the left sidebar, open the **Chat** view (or a Copilot icon, depending on VS Code version)
- Or use the Command Palette: `Ctrl+Shift+P` → search “Copilot Chat”

Examples:
- “Explain what this function does in simple words.”
- “Why am I getting this error? Here is the traceback.”
- “Suggest a minimal fix without changing behavior.”

### C) Ask Copilot to draft code changes

Examples:
- “Add input validation to the Streamlit upload step.”
- “Refactor this to be more readable for beginners.”

---

## 5) Good prompting (simple patterns that work)

Think of prompting like giving a clear experimental protocol request.

### Pattern 1: Goal + constraints

“Goal: add a new parameter to the UI.
Constraints: keep it beginner-friendly; do not add new pages; update docs.”

### Pattern 2: Provide an example input/output

“Input: a CSV with columns A,B,label.
Output: a cleaned DataFrame with missing values removed.
Please write a function with type hints.”

### Pattern 3: Ask for a checklist

“Give me a checklist to debug why Streamlit won’t start in WSL.”

---

## 6) Copilot safety + privacy for research

- Avoid pasting sensitive patient data or private datasets into AI chats.
- If you must share data-like content, consider anonymizing it first.
- Review generated code carefully before running it.

---

## 7) Using Copilot with this repo (recommended)

Useful prompts specific to this project:
- “Where is the main workflow entry point, and how does data flow through the stages?”
- “Help me add a small docstring and type hints to this function.”
- “I want to change the output folder naming; where should I do it?”

When Copilot suggests big changes, ask it to do *smaller steps*:
- “Make the smallest change that fixes the bug.”
- “Do not refactor unrelated files.”

### Practical VS Code workflow (recommended)

1. Open the file you want to change
2. Select a small block of code
3. Ask Copilot Chat: “Explain this in simple words”
4. Then ask: “Suggest a minimal improvement”
5. Apply the change and run the UI/workflow to confirm it still works

---

## 8) If Copilot suggestions are confusing

Try:
- “Explain your suggestion step-by-step like I’m new to Python.”
- “Show a minimal example.”
- “What are the risks of this change?”
