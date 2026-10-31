# Disable printing each command before it is run
.SILENT:
.PHONY: all clean

all:
	python3 -m venv .venv
	.venv/bin/python3 -m pip install -r requirements.txt
	echo "Python venv successfully created!"

clean:
	rm -rf .venv