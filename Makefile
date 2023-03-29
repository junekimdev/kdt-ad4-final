.PHONY: all up down

all: up

NAME=tensorboard

up:
	@docker compose up -d \
	&& sleep 5 \
	&& docker logs -t --tail 5 $(NAME)

down:
	docker-compose down
