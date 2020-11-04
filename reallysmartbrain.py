from imageai.Prediction import ImagePrediction
import os
execution_path=os.getcwd()

#aquí haremos la carga del modelo:
prediction = ImagePrediction()
prediction.setModelTypeAsSqueezeNet()
prediction.setModelPath( execution_path + "\squeezenet_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()

#Aquí haremos las predicciones:
predictions, probabilities = prediction.predictImage(os.path.join(execution_path,'giraffe.jpg'), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : " , eachProbability)