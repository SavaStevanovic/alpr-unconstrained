
import sys
import numpy as np
import cv2
import argparse
from tensorflow import keras
from random import choice
from sklearn.model_selection import train_test_split 

from callbacks.get_callbacks import get_callbacks

from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_network
from src.label import readShapes
from src.loss import custom_loss
from src.utils import image_files_from_folder, show
from src.data_generator import DataGenerator

from pdb import set_trace as pause

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

	model, model_stride = load_network(model_path,dim)

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


	train_data, validation_data = train_test_split(Data, test_size = 0.2, random_state = 42, shuffle = True)

	print('%d images with labels found' % len(Data))

	train_data_generator = DataGenerator(data=train_data, batch_size=4, dim=dim, model_stride=model_stride)
	validation_data_generator = DataGenerator(data=validation_data, batch_size=4, dim=dim, model_stride=model_stride)
	model_path_final  = '%s' % (model_path)

	model.fit(x = train_data_generator,validation_data = validation_data_generator , epochs=100, callbacks=get_callbacks())

	print('Stopping data generator')

	print('Saving model (%s)' % model_path_final)
	# save_model(model,model_path_final)
