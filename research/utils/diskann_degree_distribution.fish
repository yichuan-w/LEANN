#! /bin/fish

# get the dir of this script
set -x SCRIPT_DIR (dirname (realpath $0))

g++ $SCRIPT_DIR/analyze_diskann_graph.cpp -o $SCRIPT_DIR/analyze_diskann_graph

# get args
set -x INDEX_PATH $argv[1]

./analyze_diskann_graph $INDEX_PATH $INDEX_PATH.degree_distribution.txt

python plot_degree_distribution.py $INDEX_PATH.degree_distribution.txt

rm $INDEX_PATH.degree_distribution.txt