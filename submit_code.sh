#!/bin/bash

if [ "$#" -ne 2 ]; then
        echo "Usage: /scratch/eecs471w25_class_root/eecs471w25_class/$USER/final_project/submit_code.sh"
        exit 1
fi

chmod 700 new-forward.cuh
cp -f new-forward.cuh /scratch/eecs471w25_class_root/eecs471w25_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"amrhuss":rwx /scratch/eecs471w25_class_root/eecs471w25_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"aryanj":rwx /scratch/eecs471w25_class_root/eecs471w25_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"reetudas":rwx /scratch/eecs471w25_class_root/eecs471w25_class/all_sub/$USER/final_project/$USER.cuh