# handwritten-digit-recognition
A project with purpose of studying deep learning

1. Set up the environment:
- Python 3.6:
  Link download: https://www.python.org/downloads/
- Tensorflow
  Link: https://www.tensorflow.org/install/
  
  or if you use Windows, just run this command:
  > pip3 install --upgrade tensorflow
  
- skimage
  Link: http://scikit-image.org/docs/dev/install.html
  
  or if you use Windows, just run this command:
  > pip install scikit-image
  
2. Train model and test:
- Train model
  + Create a file in folder 'trains', let call it 'model_name.py'
  + Add code to build your model
  + Run command
  > python ./main.py train model_name
  (CNN model is available)
- Test
  + Put your image in folder 'data' (or any folder you want), let call it 'image_name.png'
  + Run command
  > python ./main.py test model_name path/to/image_name.png
  
Enjoy it!
