#!/bin/bash

result="./Result"
folders=`ls $result`
files=`find $result -name '*.ply*'`

for folder in $folders
do
	labelFolder="$result/$folder"

	if [ -d "${labelFolder}/txt" ]; then
		echo "${labelFolder} exists"
	else
		echo "creating dir: $labelFolder/txt"
		mkdir "$labelFolder/txt"
	fi

	ply_files=`find ${labelFolder}/ply -name '*.ply'`	
	
	for ply_file in $ply_files
	do
		txt_name=${ply_file%.ply}
		txt_name=${txt_name##*/}
		output_name="${labelFolder}/txt/${txt_name}.txt"
		sed '0,/^.*end_header/d' $ply_file|awk '{print $1" "$2" "$3" "$7" "$8" "$9}' > $output_name
		echo "Successfully writing file ${output_name}"
	done
done

#echo $files

