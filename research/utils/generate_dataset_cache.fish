#!/usr/bin/env fish

# Set default parameters
set domain "rpj_wiki"
set embedder "facebook/contriever-msmarco"
set k 5
set tasks "nq" "trivia" "hotpot" "gpqa"

# Parse command line arguments
for i in (seq 1 (count $argv))
    switch $argv[$i]
        case "--domain"
            set domain $argv[(math $i + 1)]
        case "--embedder"
            set embedder $argv[(math $i + 1)]
        case "--k"
            set k $argv[(math $i + 1)]
        case "--tasks"
            set j (math $i + 1)
            set tasks
            while test $j -le (count $argv) && not string match -q -- "--*" $argv[$j]
                set -a tasks $argv[$j]
                set j (math $j + 1)
            end
    end
end

echo "Running with the following parameters:"
echo "Domain: $domain"
echo "Embedder: $embedder"
echo "k: $k"
echo "Datasets: $tasks"

# Create directory for results
set results_dir "retrieval_results"
mkdir -p $results_dir

# Process each dataset using retrieval_demo directly
for task in $tasks
    echo ""
    echo "===== Processing dataset: $task ====="
    
    # Step 1: Run retrieval_demo with flat index to generate cache and get results
    echo "Running retrieval for $task..."
    echo "python demo/main.py --domain $domain --task $task --search --load flat --lazy"
    python demo/main.py --domain $domain --task $task --search --load flat --lazy
    
    # Check if successful
    if test $status -ne 0
        echo "Retrieval for $task failed"
        continue
    end
    
    echo "Completed processing for $task"
    echo "--------------------------------"
end

echo "All operations completed successfully!"
echo "The cache files have been created at the locations specified by get_flat_cache_path() in config.py"
echo "You can now use test_all_datasets.py to view the results" 
