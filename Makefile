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
	python3 -m unittest discover -s $(TEST_PATH) -t $(TEST_PATH)

.PHONY: requirements
requirements:
	python3 -m pip download -r ./requirements.txt -d ./requirements --no-deps
	for filename in ./requirements/*.tar.gz; do \
		echo $$filename; \
		mv $$filename $$filename.whl; \
	done;
	for filename in ./requirements/*.zip; do \
		echo $$filename; \
		mv $$filename $$filename.whl; \
	done;

.PHONY: commit
commit:
	echo $$m
	git commit -m $$m
	kaggle datasets version -p . -m $$m