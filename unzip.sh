for zip_file in LLMGraph/tasks/general/data/DyTAG_processed/*.zip; do
    unzip -n "$zip_file" -d LLMGraph/tasks/general/data
done
