all: build run

build:
	javac *.java

run:
	java Main -v

summary:
	java Main

clean:
	rm *.class