import numpy as np
import pickle
loaded_model = pickle.load(open('C:/Users/mouni/Downloads/Machine_model_deployment/trained_model.sav', 'rb'))
input_data = (2,74,0,0,0,0,0.102,22)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
