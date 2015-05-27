CC=g++

CFLAGS=-Wall -O2
OPENCV=`pkg-config opencv --cflags --libs`
LIBS=$(OPENCV)
BIN_DIR=../bin

PROGNAME=merge

all: prep
	$(CC) $(CFLAGS) main.cpp -o $(BIN_DIR)/$(PROGNAME) $(LIBS)

prep:
	mkdir -p $(BIN_DIR)

clean:
	rm -f $(BIN_DIR)