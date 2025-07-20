#!/bin/bash

# Clippy AdaGrad - GitHub Push Script
# This script prepares and pushes the project to GitHub

echo "🚀 Preparing Clippy AdaGrad for GitHub push..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "✅ No changes to commit"
else
    # Commit changes
    echo "💾 Committing changes..."
    git commit -m "Initial commit: Comprehensive Clippy AdaGrad implementation

- Complete Clippy AdaGrad optimizer implementation
- Multiple multitask learning architectures (Shared Bottom, NCF, LR)
- Professional GitHub repository structure
- Comprehensive documentation and testing
- CI/CD pipeline with GitHub Actions
- Real-world evaluation on AliExpress dataset

Inspired by: https://github.com/ledmaster/clippy-adagrad.git"
fi

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🔗 Adding remote origin..."
    git remote add origin https://github.com/WeiHanTu/Multitask_E-commerce_Recommender_System.git
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push -u origin main

echo "✅ Successfully pushed to GitHub!"
echo "🌐 Your repository is now available at: https://github.com/WeiHanTu/Multitask_E-commerce_Recommender_System.git"
echo ""
echo "📋 Next steps:"
echo "1. Enable GitHub Actions in your repository settings"
echo "2. Set up branch protection rules"
echo "3. Create your first release"
echo "4. Share your repository with the community!" 