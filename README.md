# TextDetect
Text detection in medical images.

The model currently used is trained mostly on breast screening images.

Dependencies:

 * Tensorflow
 * Keras
 * OpenCV

```bash
wget http://lawebdegalindo.com/wp-content/uploads/2012/10/mamo_dicom_p.jpg

conda create -y --name textdetect python=3.6 tensorflow-gpu keras py-opencv
conda activate textdetect

./textdetect.py mamo_dicom_p.jpg -f 2> /dev/null

conda deactivate
```
