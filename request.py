import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'e1/do':2, 'e2/do':9, 'fu/fy':6, 'type' : 0})

print(r.json())