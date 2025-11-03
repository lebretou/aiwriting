# GitHub Setup Instructions

Your repository is initialized and ready to push to GitHub!

## 📋 Steps to Push to GitHub

### 1. Create a New Repository on GitHub

Go to [github.com/new](https://github.com/new) and create a new repository:
- **Repository name**: `uncertainty-chat-app` (or your preferred name)
- **Description**: Per-Claim Uncertainty Estimation Chat with NLI
- **Visibility**: Public or Private (your choice)
- **⚠️ IMPORTANT**: Do NOT initialize with README, .gitignore, or license (we already have these)

### 2. Link Your Local Repository to GitHub

After creating the GitHub repository, run these commands:

```bash
# Add the remote repository (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/uncertainty-chat-app.git

# Or if you prefer SSH:
git remote add origin git@github.com:USERNAME/uncertainty-chat-app.git

# Verify the remote was added
git remote -v
```

### 3. Push Your Code

```bash
# Push to GitHub
git push -u origin main
```

## 🔐 Note About .env File

The `.env` file containing your OpenAI API key is **already excluded** from git via `.gitignore`. It will NOT be pushed to GitHub, keeping your API key secure.

If others clone this repository, they'll need to create their own `.env` file with:
```
OPENAI_API_KEY=their-key-here
```

## 📝 What's Already Committed

All project files are committed:
- ✅ Python source code (app.py, utils.py)
- ✅ Context documents (truth and fault-injected)
- ✅ Evidence database
- ✅ Setup scripts and documentation
- ✅ Requirements and configuration files
- ❌ .env file (excluded for security)

## 🔄 Future Updates

After making changes, commit and push with:

```bash
git add .
git commit -m "Description of your changes"
git push
```

## 📦 Current Commit

```
ba0c15f Initial commit: Per-Claim Uncertainty Chat Prototype
```

Your repository is ready to be pushed to GitHub!
