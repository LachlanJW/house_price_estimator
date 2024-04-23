import requests

response = requests.get("http://localhost:8000/houses/")

try:
    response = requests.get("http://localhost:8000/houses/")
    if response.status_code == 200:
        # Extract the JSON data from the response
        houses_data = response.json()
        for house in houses_data:
            print(f"House ID: {house['id']}")
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
except requests.RequestException as e:
    print("An error occurred while making the request:", e)
