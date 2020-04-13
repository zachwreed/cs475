
# ** Zachary Reed
# ** CS-475
# ** Makefile for proj0.cpp
# *********************************************

CXX = g++	#enables gcc as compiler
CXXFLAGS = -O3 -lm -fopenmp
DEPS = proj0

#target: dependencies
#	action

# Executable output, dependencies, and action ***************************
all: $(DEPS)

$(DEPS): $(DEPS).o
	$(CXX) $(DEPS).o -o $(DEPS) $(CXXFLAGS)

#tells compiler to create object without executable when modified
$(DEPS).o: $(DEPS).cpp
	$(CXX) -c -g $(DEPS).cpp $(CXXFLAGS)


#Removes object files and output ****************************************
clean:
	rm *.o $(DEPS)