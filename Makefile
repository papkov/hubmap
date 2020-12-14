TEST_PATH=./tests
MODULES_PATH=./modules
TRAIN_PATH=train.py inference.py

.PHONY: format lint test commit

format:
	isort $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	black $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)

lint:
	isort -c $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	black --check $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	mypy $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)

test:
	python3 -m unittest discover -s $(TEST_PATH) -t $(TEST_PATH)

requirements:
	python3 -m pip download -r ./requirements.txt -d ./requirements --no-deps

commit:
	kaggle datasets version -m "$$m" -r tar