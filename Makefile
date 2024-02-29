format: ## Run pre-commit hooks
	pre-commit run -a

sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main
