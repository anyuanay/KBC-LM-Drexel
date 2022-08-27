This is the README file describing 
    1. the code structure, and
    2. instructions for how to run the system.

1. Code Structure:
    Our submission consists of a list of Python files, a README file, and a data/ subfolder. 
    They are all located in the same directory:
        main.py: the main entry
        MyTools.py: prompts and other middle processes
        Processors.py: optimizing parameters such as top_k and thresholds
        MyHelpers.py: help functions on some logics
        baseline.py: Script provided by the organizers; called by our program
        file_io.py: Script provided by the organizers; called by our program
        README.txt (this file)
        data/
            test.jsonl
            predictions.jsonl


2. How to run the system:
    a. Place all the python files in the same directory.
    b. Create a sub-directory data/ under the directory and copy the test.jsonl to the data/ sub-directory.
    c. Open a terminal in the directory (where main.py is).
    d. Execute python ./main.py.

    The file “predictions.jsonl” will be created in the sub-directory data/.

