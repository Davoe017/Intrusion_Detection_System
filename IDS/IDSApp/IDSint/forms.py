# mlmodel/forms.py
from django import forms

class PredictionForm(forms.Form):
    dst_bytes = forms.FloatField(label='Destination Bytes')
    flag = forms.FloatField(label='Flag')
    dst_host_same_srv_rate = forms.FloatField(label='Destination Host Same Service Rate')
    count = forms.FloatField(label='Count')
    dst_host_diff_srv_rate = forms.FloatField(label='Destination Host Different Service Rate')
    service = forms.FloatField(label='Service')
    dst_host_count = forms.FloatField(label='Destination Host Count')
    protocol_type = forms.FloatField(label='Protocol')
    srv_count = forms.FloatField(label='Service Count')
