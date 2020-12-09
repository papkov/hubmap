TEST_PATH=./tests
MODULES_PATH=./modules
TRAIN_PATH=train.py inference.py

.PHONY: format
format:
	isort $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	black $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)

.PHONY: lint
lint:
	isort -c $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	black --check $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	mypy $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)

.PHONY: test
test:
	python -m unittest discover -s $(TEST_PATH) -t $(TEST_PATH)