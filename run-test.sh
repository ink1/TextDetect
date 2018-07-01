#!/bin/bash

wget http://lawebdegalindo.com/wp-content/uploads/2012/10/mamo_dicom_p.jpg

#conda create -y --name textdetect --file package-list.txt
#conda activate textdetect

./textdetect.py mamo_dicom_p.jpg -f

#conda deactivate
