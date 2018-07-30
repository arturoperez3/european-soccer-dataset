import os
import struct
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Adam Learning rate
learning_rate = 0.0001

# This method will run predictions using Tensorflow provided estimators and a custom estimator
# @param: train_a, train_b      denotes the range of data samples to use for training
# @param: test_a, test_b        denotes the range of data samples to use for testing
# @param: batchSize             size of the minibatch 
# @param: acceptableRange       the range for which the prediction can vary from the label
# @param: n_classes             a player's skill falls in the range [0,100)
def soccerPredictions (train_a, train_b, test_a, test_b, batchSize, acceptableRange, n_classes=100) :

    global tf # takes care of free variable error

    # connect to the sql database 
    connection = sqlite3.connect("/Users/Arturo1/Desktop/soccer/database.sqlite")

    #   define our features
    features_list = ["potential", "crossing", "finishing", "heading_accuracy", "short_passing", "volleys", "dribbling",
                                "free_kick_accuracy", "long_passing", "ball_control", "acceleration", "sprint_speed", "agility", "reactions",
                                "balance", "shot_power", "jumping", "stamina", "strength", "long_shots", "interceptions", "positioning",
                                "vision", "penalties", "marking", "standing_tackle", "sliding_tackle"]

    #   read in the data from the sql database
    data = pd.read_sql("""SELECT overall_rating, potential, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling,
                                free_kick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions,
                                balance, shot_power, jumping, stamina, strength, long_shots, interceptions, positioning,
                                vision, penalties, marking, standing_tackle, sliding_tackle
                                FROM Player_Attributes
                                """, connection)

    #   get rid of all empty rows/columns
    data = data.dropna()

    #   get rid of all empty rows/columns
    data = data.values

    #   seperate our data into X and Y buffers
    X = data[:, 1:]
    Y = data[:, 0]

    #   define a dictionary for later use in defining feature columns
    def getData (a,b, truncate=False):
        data_dictionary = { 
                        "potential": data[a:b,1],
                        "crossing" : data[a:b,2],
                        "finishing": data[a:b,3],
                        "heading_accuracy": data[a:b,4],
                        "short_passing": data[a:b,5],
                        "volleys": data[a:b,6],
                        "dribbling": data[a:b,7],
                        "free_kick_accuracy": data[a:b,8],
                        "long_passing": data[a:b,9],
                        "ball_control": data[a:b,10],
                        "acceleration": data[a:b,11],
                        "sprint_speed": data[a:b,12],
                        "agility": data[a:b,13],
                        "reactions": data[a:b,14],
                        "balance": data[a:b,15],
                        "shot_power": data[a:b,16],
                        "jumping": data[a:b,17],
                        "stamina": data[a:b,18],
                        "strength": data[a:b,19],
                        "long_shots": data[a:b,20],
                        "interceptions": data[a:b,21],
                        "positioning": data[a:b,22],
                        "vision": data[a:b,23],
                        "penalties": data[a:b,24],
                        "marking": data[a:b,25],
                        "standing_tackle": data[a:b,26],
                        "sliding_tackle": data[a:b,27],
        }
        #data_dictionary = dict((k, data_dictionary[k]) for k in ('sliding_tackle', 'potential', 'long_passing', 'interceptions', 'strength'
        #,'marking', 'crossing', 'finishing', 'penalties', 'stamina'))
        return data_dictionary, data[a:b,0]

    #   checks our predictions against our labels (given a certain range)
    #   for example: prediction=78, actual=75, range=5: our prediction was accurate
    #                prediction=67, actual=75, range=5: our prediction was not accurate
    def checkAccuracy(predictions_list, a, b, acceptableRange) :
        actual_list = []
        y = 0
        for y in range(a-a,b-a) :
            actual = Y[y+a,]
            actual_list.append(actual)

        ran = [False] * len(actual_list)
        x = 0 
        for x in range(a-a,b-a) :
            for i in range(0, acceptableRange+1) : 
                if (predictions_list[x]+i == actual_list[x] or predictions_list[x]-i == actual_list[x]) :
                    ran[x] = True

        percent_correct = (sum(ran) / (b-a)) * 100

        return percent_correct

    features_list_truncated = features_list
    #features_list_truncated = list(features_list[i] for i in [26, 0, 8, 20, 18, 24, 1, 2, 23, 17])

    feature_columns = [tf.feature_column.numeric_column(k) for k in features_list_truncated]

    #   input pipe for training Tensorflow estimator
    def get_input_fn(a, b, batch_size):
        # get the data
        f, labels = getData(a,b)

        # Convert the inputs to a dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(f), labels.astype(np.int64)))

        # Batch the examples
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        return dataset

    #   input pipe for evaluating Tensorflow estimator
    def eval_input_fn(a, b , batch_size):
        # get the data
        f, labels = getData(a,b)
        features=dict(f)

        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels.astype(np.int64))

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset


    #   Build a DNN with 2 hidden layers and 100 nodes in each hidden layer.
    model = tf.estimator.DNNClassifier(
        hidden_units=[150,300,200,100], feature_columns=feature_columns, n_classes=n_classes)

    #   Train the model
    model.train(
        input_fn=lambda:get_input_fn(train_a, train_b, batchSize),
        steps=5000)

    #   Evaluate the model
    e = model.evaluate(
        input_fn=lambda:eval_input_fn(test_a, test_b, batchSize))

    predictions = list(model.predict(input_fn=lambda:eval_input_fn(test_a,test_b,batchSize)))

    #   Convert our predictions to a list  
    predictions_list_tf = []
    for p in predictions:
        predictions_list_tf.append(int(p['classes'][0]))

    #   Fully connected network with Hidden Layers
    def network_withHiddenLayers(x, n_classes, reuse, isTraining) :
        with tf.variable_scope('Dense', reuse=reuse) :
            input_layer = tf.layers.dense(x, 100)
            hidden_1 = tf.layers.dense(input_layer, 150)
            hidden_1 = tf.layers.dropout(hidden_1, rate=.1, training=isTraining)
            hidden_2 = tf.layers.dense(hidden_1, 300)
            hidden_2 = tf.layers.dropout(hidden_2, rate=.1, training=isTraining)
            hidden_3 = tf.layers.dense(hidden_2, 200)
            hidden_3 = tf.layers.dropout(hidden_3, rate=.1, training=isTraining)
            hidden_4 = tf.layers.dense(hidden_3, 100)
            hidden_4 = tf.layers.dropout(hidden_4, rate=.1, training=isTraining)
            output_layer = tf.layers.dense(hidden_4, 100)
        return output_layer

    #   Define the model function
    def model_fn(features, labels, mode) :
        n_classes = 100
        output_train = network_withHiddenLayers(features, n_classes, reuse=False, isTraining=True)
        output_test = network_withHiddenLayers(features, n_classes, reuse=True, isTraining=False)

        # Predictions
        pred_test = tf.argmax(output_test, axis=1)

        print(pred_test)
        
        # Early return with our predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_test)

        # Define loss function and Adam optimizer 
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_test, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_test)

        # Define EstimatorSpec
        estimatorSpec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'predictions':pred_test},
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'actual_accuracy': acc_op})
        
        if mode == tf.estimator.ModeKeys.EVAL:
            return estimatorSpec

        return estimatorSpec


    #   Build the Estimator
    model = tf.estimator.Estimator(model_fn=model_fn)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=X[train_a:train_b, :], y=Y[train_a:train_b,],
        batch_size=batchSize, num_epochs=10, shuffle=True)

    #   Train the Model
    model.train(input_fn=input_fn)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=X[test_a:test_b, :], y=Y[test_a:test_b,],
        batch_size=batchSize, num_epochs=10, shuffle=False)

    #   Make our predictions
    p = model.predict(input_fn=input_fn)

    #   Check our accuracy and print results
    p = checkAccuracy(list(p), test_a, test_b, acceptableRange)

    tf = checkAccuracy(predictions_list_tf, test_a, test_b, acceptableRange)

    print("Custom Estimator Accuracy for Range " + str(acceptableRange) + ": " + str(p) + "%")

    print("TF Estimator Accuracy for Range " + str(acceptableRange) + ": " + str(tf) + "%")

def main() :
    train_a, train_b = 1, 100000
    test_a, test_b = 50000, 55000
    batchSize = 256
    acceptableRange = 5

    soccerPredictions(train_a, train_b, test_a, test_b, batchSize, acceptableRange)


if __name__ == "__main__":
    main()