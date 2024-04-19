#!/bin/bash

# Define the log file names for script_1.sh and script_2.sh
LOG_FILE_1="logs/trans-PL-predcls-hbt_b_t_predicate_label_lst_predicate_scores_matrix.txt"
LOG_FILE_2="logs/vctree-PL-predcls-hbt_b_t_predicate_label_lst_predicate_scores_matrix.txt"
LOG_FILE_3="logs/motifs-PL-predcls-hbt_b_t_predicate_label_lst_predicate_scores_matrix.txt"

# Function to execute scripts and log output
execute_and_log() {
    local script_name="$1"
    local log_file="$2"

    echo "Running $script_name..."
    # Redirect standard output and standard error to the log file
    ./$script_name > "$log_file" 2>&1

    echo "$script_name executed successfully."
}

# Sequentially run the .sh files in the desired order
execute_and_log "scripts/PLME/transformer/test_predcls.sh" "$LOG_FILE_1"
execute_and_log "scripts/PLME/vctree/test_predcls.sh" "$LOG_FILE_2"
execute_and_log "scripts/PLME/motifs/test_predcls.sh" "$LOG_FILE_3"

# Continue adding similar commands for other .sh files in the desired order
# For example:
# LOG_FILE_3="transformer/script_3.log"
# execute_and_log "transformer/script_3.sh" "$LOG_FILE_3"

# Add more .sh files as needed in the same manner.

# All scripts have been executed and logged
echo "All scripts executed successfully."
