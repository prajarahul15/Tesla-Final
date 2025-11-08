# Quick Git Commands Reference

## 🚀 **Initial Setup (First Time)**

```bash
# 1. Navigate to project directory
cd "E:\Tesla Model"

# 2. Initialize Git (if not already done)
git init

# 3. Check status
git status

# 4. Add all files (respects .gitignore)
git add .

# 5. Create initial commit
git commit -m "Initial commit: Tesla Financial Model Project"

# 6. Add remote repository (replace with your URL)
git remote add origin https://github.com/yourusername/tesla-financial-model.git

# 7. Push to remote
git branch -M main
git push -u origin main
```

---

## 📝 **Daily Workflow**

### **Making Changes**

```bash
# 1. Check what changed
git status

# 2. See detailed changes
git diff

# 3. Add specific files
git add path/to/file.js
git add backend/server.py
git add frontend/src/components/NewComponent.js

# Or add all changes
git add .

# 4. Commit with message
git commit -m "Add new feature: Metric forecasting auto-load"

# 5. Push to remote
git push origin main
```

---

## 🔍 **Useful Commands**

### **Viewing History**

```bash
# View commit history
git log

# One-line history
git log --oneline

# View changes in a file
git log -p path/to/file.js

# See who changed what
git blame path/to/file.js
```

### **Checking Status**

```bash
# Full status
git status

# Short status
git status -s

# See what would be committed
git diff --cached
```

### **Undoing Changes**

```bash
# Unstage a file (keep changes)
git reset HEAD path/to/file.js

# Discard changes in working directory (CAREFUL!)
git checkout -- path/to/file.js

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes - CAREFUL!)
git reset --hard HEAD~1
```

---

## 🌿 **Branching**

### **Create and Switch Branches**

```bash
# Create new branch
git checkout -b feature/new-feature

# Switch to existing branch
git checkout main

# List all branches
git branch

# Delete branch (local)
git branch -d feature/new-feature

# Delete branch (remote)
git push origin --delete feature/new-feature
```

### **Merge Branches**

```bash
# Switch to main branch
git checkout main

# Merge feature branch
git merge feature/new-feature

# Push merged changes
git push origin main
```

---

## 🔄 **Remote Operations**

### **Working with Remote**

```bash
# View remotes
git remote -v

# Add remote
git remote add origin https://github.com/username/repo.git

# Change remote URL
git remote set-url origin https://github.com/username/new-repo.git

# Remove remote
git remote remove origin
```

### **Fetch and Pull**

```bash
# Fetch changes (don't merge)
git fetch origin

# Pull changes (fetch + merge)
git pull origin main

# Pull with rebase
git pull --rebase origin main
```

---

## 📦 **Tagging Releases**

```bash
# Create a tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tags
git push origin --tags

# List tags
git tag

# Delete tag (local)
git tag -d v1.0.0

# Delete tag (remote)
git push origin --delete v1.0.0
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

```bash
# If push is rejected (remote has changes you don't have)
git pull origin main
# Resolve conflicts if any, then:
git push origin main

# If you committed wrong files
git reset HEAD~1
# Fix files, then:
git add .
git commit -m "Fixed commit"

# If you want to discard all local changes
git reset --hard origin/main

# If you want to see what's different
git diff origin/main
```

---

## 📋 **Quick Checklist Before Push**

```bash
# 1. Check status
git status

# 2. Review what will be committed
git diff --cached

# 3. Ensure no sensitive files
git status | grep -E "\.env|node_modules|venv|chroma_db"

# 4. Commit
git commit -m "Descriptive message"

# 5. Push
git push origin main
```

---

## 🎯 **Recommended Workflow**

1. **Before starting work:**
   ```bash
   git pull origin main
   ```

2. **Make changes:**
   - Edit files
   - Test changes

3. **Review changes:**
   ```bash
   git status
   git diff
   ```

4. **Stage and commit:**
   ```bash
   git add .
   git commit -m "Clear description of changes"
   ```

5. **Push:**
   ```bash
   git push origin main
   ```

---

## 📚 **Best Practices**

1. ✅ **Commit often** - Small, logical commits
2. ✅ **Write clear messages** - Describe what and why
3. ✅ **Pull before push** - Avoid conflicts
4. ✅ **Review before commit** - Check `git diff`
5. ✅ **Don't commit secrets** - Use `.env` files
6. ✅ **Don't commit large files** - Use Git LFS if needed
7. ✅ **Use branches** - For features and experiments
8. ✅ **Keep main stable** - Test before merging

---

## 🔐 **Security Reminders**

- ❌ Never commit `.env` files
- ❌ Never commit API keys or secrets
- ❌ Never commit `node_modules/` or `venv/`
- ❌ Never commit user uploads
- ✅ Use `env.example` with placeholders
- ✅ Review `git status` before committing

---

**Need help?** Check `GIT_PUBLISHING_GUIDE.md` for detailed instructions!


