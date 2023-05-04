#!/bin/bash

output_dir="./abbpcd2"

if [ -d "${output_dir}" ];then
	echo "Raw data generating folder ${output_dir} doesn't exist, creating"
	mkdir ${output_dir}
else
	echo "folder ${output_dir} exists"
fi

labels=(
  "piab_box"
  "envelope"
  "ipad_cover"
  "dustpan"
  "pencil_box"
  "white_box"
)

#folders=`ls ./Result/*`
for i in `seq 10`
do
	mkdir -p "${output_dir}/cluster${i}"
	for label in ${labels[@]}
	do
		item=`find ./Result -name "${label}_${i}.txt"`
		cp -rf $item "${output_dir}/cluster${i}"
		echo "$item copy done"
	done
	echo "cluster${i} completed"
done


