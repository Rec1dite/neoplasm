all: build run

build:
	javac *.java

run:
	java Main -v

clean:
	rm *.class