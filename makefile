all: build run

build:
	javac *.java

run:
	java Main

clean:
	rm *.class