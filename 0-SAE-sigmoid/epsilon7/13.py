from keras import optimizers, initializers
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.model_selection import KFold
import keras.backend as K
import argparse, os, time, numpy

#______________________________________________________________________#

def f1score(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	f1_score = (2*tp)/(2*tp+fp+fn+K.epsilon())

	return f1_score
#______________________________________________________________________#

def precision(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	precision = tp/(tp+fp+K.epsilon())

	return precision
#______________________________________________________________________#

def sensitivity(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	sensitivity = tp/(tp+fn+K.epsilon())

	return sensitivity
#______________________________________________________________________#

def specificity(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	specificity = tn/(tn+fp+K.epsilon())

	return specificity
#______________________________________________________________________#

def load_data(input_file):
	print('Importing file:')
	print(os.path.abspath(input_file))
	dataset = numpy.loadtxt(input_file, delimiter=',')
	X = dataset[:,0:len(dataset[0])-1]
	Y = dataset[:,len(dataset[0])-1]

	return X, Y
#______________________________________________________________________#

def set_model():
	he     = initializers.glorot_uniform(seed=1)

	inputs  = Input(shape=(340,))
	encoded = Dense(272, activation='sigmoid', kernel_initializer=he, use_bias=True, bias_initializer='zeros')(inputs)
	encoded = Dense(204, activation='sigmoid', kernel_initializer=he, use_bias=True, bias_initializer='zeros')(encoded)
	encoded = Dense(136, activation='sigmoid', kernel_initializer=he, use_bias=True, bias_initializer='zeros')(encoded)
	encoded = Dense(68,  activation='sigmoid', kernel_initializer=he, use_bias=True, bias_initializer='zeros')(encoded)
	encoded = Dense(1,   activation='sigmoid', kernel_initializer=he, use_bias=True, bias_initializer='zeros')(encoded)

	model = Model(inputs=inputs, outputs=encoded)

	adagrad = optimizers.Adagrad(lr=0.0256, epsilon=1e-7, decay=1e-13)
	model.compile(optimizer=adagrad,
			  	  loss     ='mean_squared_error',
			  	  metrics  =['accuracy', sensitivity, specificity, precision, f1score])

	return model
#______________________________________________________________________#

def train_evaluate(model, X_train, Y_train, X_test, Y_test):
	cv_scores = []

	model.fit(X_train, Y_train, epochs=256, batch_size=128, verbose=0)
	scores = model.evaluate(X_test, Y_test, verbose=0)

	print("%s: %.2f%%\t%s: %.2f%%\t%s: %.2f%%\t"
	      "%s: %.2f%%\t%s: %.2f%%\t%s: %.2f%%" % (model.metrics_names[0], scores[0]*100,
		  										  model.metrics_names[1], scores[1]*100,
									    		  "se",  scores[2]*100,
										          "spc", scores[3]*100,
												  "pre", scores[4]*100,
												  "f1s", scores[5]*100))

	for score in scores:
		cv_scores.append(score*100)

	return cv_scores
#______________________________________________________________________#

def main(args):
	numpy.random.seed(1)
	loss = []
	accu = []
	sens = []
	spec = []
	prec = []
	f1sc = []

	X, Y = load_data(args.input_file)
	kFold = KFold(n_splits=10)
	for train, test in kFold.split(X, Y):
		model = None
		model = set_model()
		cv_scores = train_evaluate(model, X[train], Y[train], X[test], Y[test])

		loss.append(cv_scores[0])
		accu.append(cv_scores[1])
		sens.append(cv_scores[2])
		spec.append(cv_scores[3])
		prec.append(cv_scores[4])
		f1sc.append(cv_scores[5])

	with open('results.txt', 'a') as file_results:
		file_results.write('------------------------------------------------------------------------------------------\n')
		for i in range(0, len(accu)):
			file_results.write('loss: '   + format(loss[i], '.2f') + '\tacc: ' + format(accu[i], '.2f') +
			                   '\tse: '   + format(sens[i], '.2f') + '\tspc: ' + format(spec[i], '.2f') +
							   '\tprec: ' + format(prec[i], '.2f') + '\tf1s: ' + format(f1sc[i], '.2f') + '\n')
		file_results.write('==========================================================================================\n')
		file_results.write('loss: ' + format(numpy.mean(loss), '.2f') + ' (+/- ' + format(numpy.std(loss), '.2f') + ')\n')
		file_results.write('acc: '  + format(numpy.mean(accu), '.2f') + ' (+/- ' + format(numpy.std(accu), '.2f') + ')\n')
		file_results.write('se:  '  + format(numpy.mean(sens), '.2f') + ' (+/- ' + format(numpy.std(sens), '.2f') + ')\n')
		file_results.write('spc: '  + format(numpy.mean(spec), '.2f') + ' (+/- ' + format(numpy.std(spec), '.2f') + ')\n')
		file_results.write('pre: '  + format(numpy.mean(prec), '.2f') + ' (+/- ' + format(numpy.std(prec), '.2f') + ')\n')
		file_results.write('f1s: '  + format(numpy.mean(f1sc), '.2f') + ' (+/- ' + format(numpy.std(f1sc), '.2f') + ')\n')
		file_results.write('------------------------------------------------------------------------------------------\n')
		file_results.write('\n\n')

	print("loss:  %.2f%% (+/- %.2f%%)\n"
		  "acc:   %.2f%% (+/- %.2f%%)\n"
		  "se:    %.2f%% (+/- %.2f%%)\n"
		  "spc:   %.2f%% (+/- %.2f%%)\n"
		  "pre:   %.2f%% (+/- %.2f%%)\n"
	  	  "f1s:   %.2f%% (+/- %.2f%%)" % (numpy.mean(loss), numpy.std(loss),
		  								  numpy.mean(accu), numpy.std(accu),
										  numpy.mean(sens), numpy.std(sens),
										  numpy.mean(spec), numpy.std(spec),
	  						   			  numpy.mean(prec), numpy.std(prec),
										  numpy.mean(f1sc), numpy.std(f1sc)))
#______________________________________________________________________#

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='')

	parser.add_argument('-in', action='store', dest='input_file', required=True,
						 help='Input file in dataset training format.')

	start_time = time.time()
	args = parser.parse_args()
	print('Starting process...')
	main(args)
	minutes = (time.time() - start_time) / 60
	seconds = (time.time() - start_time) % 60
	print('Process completed!\n'
          'Used time: %.0f minutes and %.d seconds' % (minutes, seconds))
