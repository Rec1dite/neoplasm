all: build run clean

build:
	javac *.java

run:
	java Main -v

clean:
	rm *.class