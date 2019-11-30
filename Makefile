develop:
	pip install -e .
install:
	pip install .
uninstall:
	pip uninstall -y passgan
clean:
	rm -rf *.egg-info
	find ./perceptor/ -name "__pycache__" -type d -prune -exec rm -rf "{}" \;

.PHONY: develop