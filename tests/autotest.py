#!/usr/bin/python

"""
(Temporary!) autotesting for the OP2 ROSE implementation

Runs op2rose on each of the test cases, executes the resulting binary
and compares outputs with a set of expected outputs.. Anomalies are highlighted, 
and the user may replace the expected output with the new output (in the case
where the change is correct).
"""

# Python modules
import sys
import os
from subprocess import Popen, PIPE, call
from shutil import copy

# A nicer traceback from IPython
from IPython import ultraTB

# For colouring diffs
from pygments import highlight
from pygments.lexers import DiffLexer
from pygments.formatters import TerminalFormatter

# List of tests - needs manually updating for each test.
# Obviously there's a better way.
TESTS = ['jac', 'airfoil']

def main():

    sys.excepthook = ultraTB.FormattedTB(mode='Context')

    if(len(sys.argv) > 1):
      tests = [sys.argv[1]]
    else:
      tests = TESTS
    
    tests.sort()

    for test in tests:
	check(test)

    sys.exit(0)

def check(test):

    print "Testing " + test

    os.chdir(test)
    call(['make','testexec'])
    
    outputfiles = os.listdir('expected')
    for output in outputfiles:
        expectedfile = 'expected/'+output
        cmd = "diff -u " + expectedfile + " " + output
        diff = Popen(cmd, shell=True, stdout=PIPE)
        diffout, differr = diff.communicate()
    
        if diffout:
            print "When testing ", test, "Difference detected in ", output, ", ",
            diffmenu(output, diffout)

    os.chdir('..')

def diffmenu(outputfile, diffout):
    
    print "[Continue, Abort, View, Replace?] ",
    response = sys.stdin.readline()
    rchar = response[0].upper()

    if rchar=='C':
        return
    elif rchar=='A':
        sys.exit(-1)
    elif rchar=='V':
        print highlight(diffout, DiffLexer(), TerminalFormatter(bg="dark"))
	diffmenu(outputfile, diffout)
    elif rchar=='R':
	dest = 'expected/'+outputfile
	copy(outputfile, dest)
    else:
        print "Please enter a valid option: ", 
	diffmenu(sourcefile, diffout)

# Execute main

if __name__ == "__main__":
    main()

