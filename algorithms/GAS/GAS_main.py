
'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import tensorflow as tf
from tensorflow.keras import optimizers
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
from algorithms.GAS.GAS import GAS
import time
from utils.data_loader import *
from utils.utils import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--train_size', type=float, default=0.2, help='training set percentage')
parser.add_argument('--dataset_str', type=str, default='example', help="['dblp','example']")
parser.add_argument('--epoch_num', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--lr', default=0.001, help='learning rate')
<<<<<<< Updated upstream
parser.add_argument('--learning_rate', default=0.001, help='the ratio of training set in whole dataset.')
=======
>>>>>>> Stashed changes

# GAS
parser.add_argument('--review_num sample', default=7, help='review number.')
parser.add_argument('--gcn_dim', type=int, default=5, help='gcn layer size.')
parser.add_argument('--output_dim1', type=int, default=64)
parser.add_argument('--output_dim2', type=int, default=64)
parser.add_argument('--output_dim3', type=int, default=64)
parser.add_argument('--output_dim4', type=int, default=64)

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

<<<<<<< Updated upstream
def main(adj_list: list, features: tf.SparseTensor, label: tf.Tensor, masks: list, args):
    model = GAS(args.input_dim_i, args.input_dim_u, args.input_dim_r, args.h_u_size, args.h_i_size,
                args.output_dim1, args.output_dim2, args.output_dim3, args.output_dim4, args.gcn_dim, args)
    optimizer = optimizers.Adam(lr=args.lr)

    for epoch in range(args.epoch_num):

        with tf.GradientTape() as tape:
            train_loss, train_acc = model([adj_list, features, label, masks[0]])

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_loss, val_acc = model([adj_list, features, label, masks[1]])
        print(f"Epoch: {epoch:d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f},"
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        test_loss, test_acc = model([adj_list, features, label, masks[2]])
        print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")


if __name__ == "__main__":
    adj_list, features, idx_train, idx_val, idx_test, y = load_data_gas(train_size=args.train_size)

    # convert to dense tensors
    label = tf.convert_to_tensor(y, dtype=tf.float32)

    # initialize the model parameters
    node_embedding_r = features[0].shape[1]
    node_embedding_u = features[1].shape[1]
    node_embedding_i = features[2].shape[1]

    args.nodes_num = features[0].shape[0]
    args.class_size = y.shape[1]
    args.input_dim_i = features[2].shape[1]
    args.input_dim_u = features[1].shape[1]
    args.input_dim_r = features[0].shape[1]
    args.h_u_size = adj_list[0].shape[1] * (node_embedding_r + node_embedding_u)
    args.h_i_size = adj_list[2].shape[1] * (node_embedding_r + node_embedding_i)

    masks = [idx_train, idx_val, idx_test]

    main(adj_list, features, label, [idx_train, idx_val, idx_test], args)
=======
## dou: features type is list of array not sparseTensor
def main(adj_list: list, features: tf.SparseTensor, label: tf.Tensor, masks: list, args):

	# dou: using one args as the input and expand all arguments in the GAS class initialization
	model = GAS(args.input_dim_i, args.input_dim_u, args.input_dim_r, args.h_u_size, args.h_i_size,
				args.output_dim1, args.output_dim2, args.output_dim3, args.output_dim4, args.gcn_dim, args)
	optimizer = optimizers.Adam(lr=args.lr)

	for epoch in range(args.epoch_num):

		with tf.GradientTape() as tape:
			train_loss, train_acc = model([adj_list, features, label, masks[0]])

		grads = tape.gradient(train_loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

		val_loss, val_acc = model([adj_list, features, label, masks[1]])
		print(f"Epoch: {epoch:d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f},"
			  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

		# dou: move the testing outside the for loop
		test_loss, test_acc = model([adj_list, features, label, masks[2]])
		print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")


if __name__ == "__main__":
	# add more comments to the functions. specify which equations the functions are corresponded to
	# dou: train val test needs to be split twice
	adj_list, features, idx_train, idx_val, idx_test, y = load_data_gas(train_size=args.train_size)

	# convert to dense tensors
	label = tf.convert_to_tensor(y, dtype=tf.float32)

	# initialize the model parameters
	# dou: following three lines have same effect comparing with the args.input_dim_i etc.
	node_embedding_r = features[0].shape[1]
	node_embedding_u = features[1].shape[1]
	node_embedding_i = features[2].shape[1]

	# dou: give a new name to nodes_num, it represents the number of reviews
	args.nodes_num = features[0].shape[0]
	args.class_size = y.shape[1]
	args.input_dim_i = features[2].shape[1]
	args.input_dim_u = features[1].shape[1]
	args.input_dim_r = features[0].shape[1]
	args.h_u_size = adj_list[0].shape[1] * (node_embedding_r + node_embedding_u)
	args.h_i_size = adj_list[2].shape[1] * (node_embedding_r + node_embedding_i)

	masks = [idx_train, idx_val, idx_test]

	main(adj_list, features, label, [idx_train, idx_val, idx_test], args)
>>>>>>> Stashed changes
