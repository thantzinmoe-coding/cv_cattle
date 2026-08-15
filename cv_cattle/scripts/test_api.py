import requests

url = "http://127.0.0.1:8000/predict/image"
files = {'file': open('../CattleLameness/Data/Lame/L (1).mp4', 'rb')}
r = requests.post(url, files=files)
print(r.status_code)
print(r.json())
