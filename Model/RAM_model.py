import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
 

# Define useful parameters:
max_abs_coordinate = 2
n_of_partitions = 28 # for one dimension
partition = np.around(max_abs_coordinate / n_of_partitions, decimals=3)
window_dim = 8 # i.e. 8x8
n_classes = 10
n_gazes = 7
loc_std = 0.15  # or  0.11 , both from github, 0.15 seems to work better 
Image_dim = 28
pad_dim = window_dim//2 # for one side
RNN_cell_size = 256 # RNN number of output cells
batch_size = 64 
n_epochs =  50000
learning_rate = 4e-3
max_grad_norm = 1 # ?
adj_loc_value = np.around(Image_dim / (Image_dim+window_dim), decimals=3) # Once add padding, need to rescale locations
																		 # so avoid centering window onto pads, this can
																		 # be done by rescaling locations by ratio of old
																		 # and new dimension (i.e. after padding) since
																		 # window will be used on padded image 




# Architecture -------------------------------------------------------------------------- 

def retina_sensor(images,location):
	
	
	images = tf.reshape(images,shape= [-1,28,28,1]) # reshape to [batch_size, width, height, channels]

	paddings = tf.constant([[0,0],[pad_dim,pad_dim],[pad_dim,pad_dim],[0,0]]) # apply padding to row and columns (before and after), but not to batch or channel
	
	images = tf.pad(images, paddings,'CONSTANT', constant_values = 0) # add padding to image based on window size

	adjusted_loc = tf.multiply(location, adj_loc_value) # prevent centering window on pads

	adjusted_loc= location
	
	window = tf.image.extract_glimpse(images,[8,8],adjusted_loc, centered=True, normalized=True)
	

	return tf.reshape(window,[-1,window_dim*window_dim]) # bring back to [batch_size, w_dim*w_dim]


def glimplse_net(image,location):
	
	
	window = retina_sensor(image,location)
	
	with tf.variable_scope('glimpse_net',reuse=tf.AUTO_REUSE):
	
		# Fully connected layers for the window and location independetly
		Logits_window = tf.contrib.layers.fully_connected(window,128, activation_fn=tf.nn.relu, scope= 'wd_nn') # input shape =[batch_size, depth], i.e. batch_size can be any
		Logits_location = tf.contrib.layers.fully_connected(location, 128, activation_fn=tf.nn.relu, scope='cl_nn')

		# Fully connected layer combining location and window together
		Logits_WindLoc = tf.concat([Logits_window,Logits_location], axis=1)
		Logits_g = tf.contrib.layers.fully_connected(Logits_WindLoc, 256, activation_fn=tf.nn.relu, scope='wl_nn')

		return Logits_g


def location_net(state):


	with tf.variable_scope('location_net',reuse=tf.AUTO_REUSE):
		
		locat = tf.contrib.layers.fully_connected(state,2, activation_fn=None, scope='l_nn')

		locat = tf.clip_by_value(locat, -1.,1.)

		return locat



def classification_net(state):

	
	with tf.variable_scope('classification_net',reuse=tf.AUTO_REUSE):

		Logit = tf.contrib.layers.fully_connected(state, n_classes, activation_fn=None, scope='a_nn')

		return Logit



def baseline(state):

	with tf.variable_scope('baseline',reuse=tf.AUTO_REUSE):

		Logit = tf.contrib.layers.fully_connected(state,1,activation_fn=None, scope='b_nn') 

		return tf.squeeze(Logit, axis=-1) # need one value for each element in the batch
										  # and fully_con output has extra dim (5,1)



def core_model(Image_placeholder, train): 

	# In this case at every images the initial state is zero and location is randomly sampled
	h_prev = tf.zeros([batch_size,RNN_cell_size], dtype=tf.float32)
	sampled_location = tf.random_uniform((batch_size,2), minval=-1, maxval=1)


	# initialise some useful variables  
	prob_location = []  
	baseline_output = []
	list_locations = [sampled_location]
	list_mean_loc = []
	list_states = []

	# Implement RNN with a for loop:
	for i in range(n_gazes):

	
		with tf.variable_scope('core_net', reuse=tf.AUTO_REUSE):
	
		
			G_output = glimplse_net(Image_placeholder, sampled_location)		
			
			# RNN:
			
			input_and_state = tf.concat([G_output,h_prev], axis = 1)
			h = tf.contrib.layers.fully_connected(input_and_state, RNN_cell_size, activation_fn=tf.nn.relu, scope='r_nn')
		
			list_states.append(h)
		
		
			# Don't compute following location & baseline for final iteration
			if i < n_gazes -1:

				baseline_values = baseline(tf.stop_gradient(h))
				baseline_output.append(baseline_values) # want a predicted reward for each USED location 
													# overal output = [ n_gazes -1, batch_size]
				h_prev = h
 
				
				# Compute location means
				mean_location = location_net(tf.stop_gradient(h))
			
				list_mean_loc.append(mean_location)
	
				# Sample locations only during training, during testing use output mean directly
				if train:
				
					# Sample from normal distrib with output mean and fixed std, for each batch sample from corresponding distrib - determined by output mean
					# Stop gradient, otherwise loc_net gets trained by cross-entropy by back-tracking from h to glimpse-net and from there to sampled-loc   
					sampled_location = tf.stop_gradient(tf.random_normal([batch_size,2], mean = mean_location, stddev= loc_std )) 
			

					sampled_location = tf.clip_by_value(sampled_location, -1., 1.)

					list_locations.append(sampled_location)

					# Create batch_size normal distribs based on mean output for each element in a batch 
					location_distribution =  tf.distributions.Normal(mean_location , loc_std)


					# Calculate log "p" for each sampled action and sum them up across 2 dim locations 
					prob_location.append(tf.reduce_sum(location_distribution.log_prob(sampled_location),axis=-1)) # this results in a list [n_gazes-1, batch_size]
					

				else:

					sampled_location = mean_location


	return h, prob_location, baseline_output, list_locations, list_mean_loc, list_states 


# Costs & train model -------------------------------------------------------------------------- 

def REINFORCE(l_probabilities, rewards, baseline_val):

	rwds = tf.tile(tf.expand_dims(rewards,1),[1,n_gazes-1]) # multiplicate rewards to have one reward for each gaze 
												   		   # output by the location NN, i.e. all but first one 
	
	# Apply baseline:
	adj_rwds = rwds - tf.transpose(tf.stop_gradient(baseline_val)) # adj_rew will be used to optimise the l_NN and don't want TF to
													 			   # backtrack to baseline_val network during backprop, transpose it since 
													 			   # baseline = [n_gazes-1, batch_size], rwds = [batch_size, n_gazes-1]
	
	REINF_cost = tf.reduce_mean(adj_rwds * tf.transpose(l_probabilities)) # l_probabilities= [n_gazes-1, batch_size] need transposing
																		   # overall mean across gazes (i.e REINFORCE cost) and batches (i.e. as usually done)																	   

	baseline_cost = tf.losses.mean_squared_error(labels= rwds, predictions = tf.transpose(baseline_val)) # overall mean squared error across gazes and batches

	return -REINF_cost, baseline_cost # REINF_cost is negative since want to maximise it   



def training_Opt(loss):

	glob_step = tf.get_variable('global_step',[], initializer=tf.constant_initializer(0), trainable= False)

	
	adapted_lr = tf.train.exponential_decay(learning_rate, global_step= glob_step, decay_steps = 1200, decay_rate = 0.95, staircase=True) 

	adapted_lr = tf.maximum(adapted_lr, learning_rate / 10)

	trainable_vars = tf.trainable_variables()

	gradients = tf.gradients(loss, trainable_vars)

	

	train_opt = tf.train.AdamOptimizer(adapted_lr)

	# clip gradient
	gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)

	train_opt = train_opt.apply_gradients(zip(gradients,trainable_vars), global_step=glob_step)

	return train_opt, adapted_lr  #[tf.norm(grad) for grad in gradients] # latter for debugging
                                       


def train_model(Image_placeholder, BatchY_placeholder):


	train = True

	h, prob_location, baseline_output, _, _, _ = core_model(Image_placeholder, train)



	# Classification:
	A_output = classification_net(h)
	prediction = tf.argmax(tf.nn.softmax(A_output), axis = 1)
	_Y = tf.argmax(BatchY_placeholder, axis=1)


	# Compute losses:

	entropy_output = tf.nn.log_softmax(A_output)

	rewards = tf.cast(tf.equal(prediction,_Y),dtype=tf.float32)

	REINF_cost, baseline_cost = REINFORCE(prob_location,tf.stop_gradient(rewards), baseline_output)

	entropy_loss = -tf.reduce_sum(tf.cast(BatchY_placeholder, dtype=tf.float32) * entropy_output, axis=1) 


	mean_entropy_loss = tf.reduce_mean(entropy_loss)

	total_loss = REINF_cost + baseline_cost + mean_entropy_loss 


	# Optimisation:
	train_step, learning_rate_adapted = training_Opt(total_loss)

	return train_step,total_loss, REINF_cost, baseline_cost, mean_entropy_loss,rewards 


# Test model -------------------------------------------------------------------------- 

def test_model(Image_placeholder, BatchY_placeholder):

	train = False

	h, _, _, _,_,_ = core_model(Image_placeholder, train)

	# Classification:
	A_output = classification_net(h)
	prediction = tf.argmax(tf.nn.softmax(A_output), axis = 1)
	_Y = tf.argmax(BatchY_placeholder, axis=1)

	# Accuracy
	Test_prediction = tf.reduce_mean(tf.cast(tf.equal(prediction,_Y),dtype=tf.float32))

	return Test_prediction




#-----------------------------------------------------------------
# Debugging method - check whether weights get updated
def print_var(values_list1,values_list2, list_names):

	
			
	for a,b,c in zip(values_list1,values_list2,list_names):

		print('Variable: ', c.name)

		p= np.equal(a, b)

		shape_ = np.array(a).shape
		
		if len(shape_) >1:
			
			print('N. of Variables: ', shape_[0]*shape_[1] )
		
		else:

			print('N. of Variables: ',shape_)

		
		print('N. of Variables equal: ', np.count_nonzero(p), '\n')

#-----------------------------------------------------------------



# Execution -------------------------------------------------------------------------- 

# Define some useful placeholder for network
Image_placeholder = tf.placeholder(tf.float32, shape=[None, Image_dim*Image_dim])
BatchY_placeholder = tf.placeholder(tf.int64, shape =[None, n_classes])

train = train_model(Image_placeholder, BatchY_placeholder)

test = test_model(Image_placeholder, BatchY_placeholder)




# Run model in a TF session: ---------------------------------------------------------

with tf.Session() as sess:


	init = sess.run(tf.global_variables_initializer())
	
	# Load MNIST dataset
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	
	# Train the model:

	batch = mnist.train.next_batch(batch_size) # extract one image

	training_images = batch[0] # extract pciture only from tuple of pictures and labels
	label_images = batch[1]

	
	
	
	for i in range(n_epochs):
	

		batch = mnist.train.next_batch(batch_size) # extract one image

		training_images = batch[0] # extract pciture only from tuple of pictures and labels
		label_images = batch[1]

		

		_, T_Loss, R_cost, b_cost, e_loss,rewards_ = sess.run(train, feed_dict={Image_placeholder: training_images, BatchY_placeholder: label_images})

		


		if i % 500 == 0:
			
			print('Total loss at {0}: '.format(i), T_Loss, '\n')
			print('REINFORCE Cost: ', R_cost)
			print('Baseline Cost: ', b_cost)
			print('Entropy Cost: ', e_loss, '\n')
			print('rewards: ', sum(rewards_), '\n')
			

	

	# Test the model:

	test_accuracy =[]
	
	for e in range(mnist.test.images.shape[0] // batch_size): # iterate for all images in the testing set

		t_batch = mnist.test.next_batch(batch_size)

		testing_images = t_batch[0]
		testing_labels = t_batch[1]

		accuracy_ = sess.run(test, feed_dict={Image_placeholder: testing_images, BatchY_placeholder: testing_labels})

		test_accuracy.append(accuracy_) # store accuracy for each batch

	print('Mean test-acc: ', np.mean(test_accuracy)) # mean accuracy across all test batches	

		
					




