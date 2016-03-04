#!/bin/bash

while [ 1 ]
do
    rsync -avz --progress --ignore-existing . gpu01:~/Image_Retrival_Engine
    if [ "$?" = "0" ] ; then
        echo "rsync completed normally"
        exit
    else
        echo "Rsync failure. Backing off and retrying..."
        sleep 10
    fi
done
