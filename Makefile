.PHONY: down up down_up attach

# Stop and remove containers
down:
	@echo "Stopping and removing containers..."
	docker compose down

# Start containers in detached mode
up:
	@echo "Starting containers..."
	docker compose up -d

# Down, then up again
down_up:
	@echo "Stopping and removing containers..."
	docker compose down
	@echo "Starting containers..."
	docker compose up -d

# Attach: down, up, and exec into the client service (replace `client` with your service name)
attach:
	@echo "Stopping and removing containers..."
	docker compose down
	@echo "Starting containers..."
	docker compose up -d
	@echo "Attaching to client service..."
	docker compose exec client /bin/bash
