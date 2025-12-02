# Detect images with faces to compare poses and facial expressions

## Setup
1. Clone this repo and navigate to this directory via `cd`.
2. In this directory, make sure you have a subdirectory called "./all_images" that contains the full image dataset.
3. Once in this directory, create a virtual environment with python 3.10.
```
python3.10 -m venv venv
source venv/bin/activate
```
3. Run necessary installations with appropriate dependencies/version matching.
```
pip install --upgrade pip setuptools wheel
pip install tensorflow==2.16.1            
pip install "tf-keras<2.17"               
pip install "numpy<2"
pip install deepface==0.0.93 --no-deps
pip install "gdown>=3.10.1" opencv-python==4.8.1.78 mtcnn retina-face Pillow ultralytics
pip install "fire>=0.4.0" "Flask>=1.1.2" "flask-cors>=4.0.1" "gunicorn>=20.1.0" "pandas>=0.23.4"
```
4. Run the face detection script.
```
python face_detection.py
```
5. The script created a new subdirectory called "./face_images" which contains the images with visible faces -- now you can use these images to get a pose classification and facial expression classification to compare the two.
