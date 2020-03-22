
import sys
import numpy as np
import cv2
import argparse
from tensorflow import keras

from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import custom_loss
from src.utils import image_files_from_folder, show
from src.data_generator import DataGenerator

from pdb import set_trace as pause


def load_network(modelpath,input_dim):

	model = load_model(modelpath)
	input_shape = (input_dim,input_dim,3)

	# Fixed input size for training
	inputs  = keras.layers.Input(shape=(input_dim,input_dim,3))
	outputs = model(inputs)

	output_shape = tuple(outputs.shape[1:])
	output_dim   = output_shape[1]
	model_stride = input_dim // output_dim

	assert input_dim % output_dim == 0, \
		'The output resolution must be divisible by the input resolution'

	assert model_stride == 2**4, \
		'Make sure your model generates a feature map with resolution ' \
		'16x smaller than the input'

	return model, model_stride, input_shape, output_shape

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-its'		,'--iterations'		,type=int   , default=300000	,help='Number of mini-batch iterations (default = 300.000)')
	parser.add_argument('-bs'		,'--batch-size'		,type=int   , default=32		,help='Mini-batch size (default = 32)')
	parser.add_argument('-od'		,'--output-dir'		,type=str   , default='./'		,help='Output directory (default = ./)')
	parser.add_argument('-op'		,'--optimizer'		,type=str   , default='Adam'	,help='Optmizer (default = Adam)')
	parser.add_argument('-lr'		,'--learning-rate'	,type=float , default=.001		,help='Optmizer (default = 0.01)')
	args = parser.parse_args()

	netname 	= basename('v1')
	train_dir 	= 'data/to_label/cars_test'
	outdir 		= args.output_dir
	model_path	= 'saved_model/'
	iterations 	= args.iterations
	batch_size 	= args.batch_size
	dim 		= 208

	if not isdir(outdir):
		makedirs(outdir)

	model,model_stride,xshape,yshape = load_network(model_path,dim)

	opt = getattr(keras.optimizers,args.optimizer)(lr=args.learning_rate)
	model.compile(loss=custom_loss, optimizer=opt)

	print('Checking input directory...')
	Files = image_files_from_folder(train_dir)

	Data = []
	for file in Files:
		labfile = splitext(file)[0] + '.txt'
		if isfile(labfile):
			L = readShapes(labfile)
			if L:
				I = cv2.imread(file)
				Data.append([I,L[0]])
	Data = np.array(Data)

	print('%d images with labels found' % len(Data))

	dg = DataGenerator(	data=Data, \
						batch_size=4, \
						dim=dim, \
						model_stride=model_stride)
	model_path_final  = '%s' % (model_path)

	model.fit(x=dg, epochs=100)

	print('Stopping data generator')

	print('Saving model (%s)' % model_path_final)
	save_model(model,model_path_final)
