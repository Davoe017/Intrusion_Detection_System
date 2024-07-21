from django.shortcuts import render
from .forms import PredictionForm
import numpy as np
import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model('C:\\Users\\David\\Desktop\\University Studies\\400 level omega semester\\CIS421\\IDSmodel.h5')

def predict(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract form data
            dst_bytes = form.cleaned_data['dst_bytes']
            flag = form.cleaned_data['flag']
            dst_host_same_srv_rate = form.cleaned_data['dst_host_same_srv_rate']
            count = form.cleaned_data['count']
            dst_host_diff_srv_rate = form.cleaned_data['dst_host_diff_srv_rate']
            service = form.cleaned_data['service']
            dst_host_count = form.cleaned_data['dst_host_count']
            protocol_type = form.cleaned_data['protocol_type']
            srv_count = form.cleaned_data['srv_count']

            # Prepare input data for prediction
            input_data = np.array([[dst_bytes, flag, dst_host_same_srv_rate, count, 
                                    dst_host_diff_srv_rate, service, dst_host_count, 
                                    protocol_type, srv_count]], dtype=np.float32)

            # Perform prediction
            prediction = model.predict(input_data)
            predicted_class = 'Normal' if prediction[0] < 0.5 else 'Anomalous'

            # Render prediction result template
            return render(request, 'IDSint/result.html', {'form': form, 'predicted_class': predicted_class})
    else:
        form = PredictionForm()

    return render(request, 'IDSint/form.html', {'form': form})
