
# PROGEX (Program Graph Extractor)

version: 3.4.5 (October 2019)

PROGEX is a cross-platform tool for extracting well-known graphical program representations
from software source code. PROGEX is written in Java, based on the ANTLR parser generator.

PROGEX reads program source code files as input, and is able to generate various
graphical program representations and export them into well-known file formats for graphs;
such as DOT, GML, and JSON.

## Usage Guide

```
USAGE:

   java -jar PROGEX.jar [-OPTIONS...] /path/to/program/src

OPTIONS:

   -help      Print this help message
   -outdir    Specify path of output directory
   -format    Specify output format; either 'DOT', 'GML', or 'JSON'
   -lang      Specify language of program source codes

   -ast       Perform AST (Abstract Syntax Tree) analysis
   -cfg       Perfomt CFG (Control Flow Graph) analysis
   -icfg      Perform ICFG (Interprocedural CFG) analysis
   -info      Analyze and extract detailed information about program source code
   -pdg       Perform PDG (Program Dependence Graph) analysis

   -debug     Enable more detailed logs (only for debugging)
   -timetags  Enable time-tags and labels for logs (only for debugging)

DEFAULTS:

   - If not specified, the default output directory is the current working directory.
   - If not specified, the default output format is DOT.
   - If not specified, the default language is Java.
   - There is no default value for analysis type.
   - There is no default value for input directory path.

EXAMPLES:

   java -jar PROGEX.jar -cfg -lang java -format gml  /home/user/project/src

      This example will extract the CFG of all Java source files in the given path and 
      will export all extracted graphs as GML files in the current working directory.

   java -jar PROGEX.jar -outdir D:\outputs -pdg  C:\Project\src

      This example will extract the PDGs of all Java source files in the given path and 
      will export all extracted graphs as DOT files in the given output directory.

NOTES:

   - The important pre-assumption for analyzing any source code is that the 
     program is valid according to the grammar of that language. Analyzing 
     invalid programs has undefined results; most probably the program will 
     crash!

   - Analyzing large programs requires high volumes of system memory, so 
     it is necessary to increase the maximum available memory to PROGEX.

     In the example below, the -Xmx option of the JVM is used to provide PROGEX 
     with 5 giga-bytes of system memory; which is required for the PDG analysis 
     of very large programs (i.e. about one million LoC). Needless to say, this 
     is possible on a computer with at least 8 giga-bytes of RAM:

        java -Xmx5G -jar PROGEX.jar -pdg ...
```


## Installation and Requirements

PROGEX is a fully portable tool and requires no installation.
Installing a Java Runtime Environment (JRE, version 8 or newer) is the only requirement for running PROGEX.
To acquire the latest JRE version for your platform, visit https://java.com/


## Visualizing Output Graphs

PROGEX can export the extracted graphs into DOT format. 
This format can be visualized using the `xdot` program.
To install `xdot` on Ubuntu, use the following command:

`sudo apt install xdot`

And visualize the resulting graph as below:

`xdot graph.dot`

An alternative way is to create an image file. This can be done as follows:

`dot -Tpng -o graph.png graph.dot`
