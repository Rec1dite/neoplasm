all: build run

build:
	javac *.java

run:
	java Main -v

summary:
	java Main

help: build
	java Main -h

clean:
	rm *.class