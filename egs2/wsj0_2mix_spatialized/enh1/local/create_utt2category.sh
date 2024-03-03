#!/bin/bash

wavscp_path="$1"

output_dir=$(dirname "$wavscp_path")
utt2category_path="$output_dir/utt2category"

while IFS= read -r line; do
    parts=($line)
    id="${parts[0]}"
    path="${parts[1]}"

    if [[ $id == *"reverb"* ]]; then
        category="reverb"
    elif [[ $id == *"anechoic"* ]]; then
        category="anechoic"
    fi

    echo -e "$id $category" >> "$utt2category_path"
done < "$wavscp_path"
