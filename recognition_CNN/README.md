**README**

Author: Luca Scimeca

Recognition neural networks for WRS robotics challenge, Tokyo 2018.

Dependencies:
- Python 3.6
- tensorflow-gpu
- CuDNN + Cuda for gpu computations 
- cv2


WORKFLOW:

1) You can gather data like you would for imagenet, with the images in folder "data/imagenet_data/train_set/images" 
	and respective labels in "data/imagenet_data/train_set/labels". The labels must be xml files logging bounding boxes 
	and classes for each  object in the images. You can label the data through the tool in the "labelImg" folder.
2) Run "imagenet_to_tensorflow.py" script to create a tensorflow dataset, with each object cropped out and padded from 
	original images. Modify the script to change data input/output paths.
3) Use the "tensorflow_to_numpy.py" script to generate a numpy ".npz" file containing the dataset to 	train the network on. 
	Modify the script to change data input/output paths.
4) Run myCNN.py or myCNNdeep.py to train a network on the generated numpy dataset.
5) Can oberve training with "tensorboard --logdir=/dir-to-trainsummaries"

6) After training freeze model, e.g. "python freeze_model.py --model_dir out-deepcnn/checkpoints --output_node_names "inputs,predictions/Softmax"

7) Can test the model in test.py



all code, except for "labelImg", and part of the mlp-set up was created ad-oc for this application.
