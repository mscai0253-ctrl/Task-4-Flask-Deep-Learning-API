import requests


BASE_URL = "http://127.0.0.1:5000"


def test_home():
    response = requests.get(BASE_URL + "/")

    print("\n--- HOME TEST ---")
    print("Status Code:", response.status_code)
    print("Response:", response.json())


def test_health():
    response = requests.get(BASE_URL + "/health")

    print("\n--- HEALTH TEST ---")
    print("Status Code:", response.status_code)
    print("Response:", response.json())


def test_prediction():
    data = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    response = requests.post(
        BASE_URL + "/predict",
        json=data
    )

    print("\n--- PREDICTION TEST ---")
    print("Status Code:", response.status_code)
    print("Response:", response.json())


def test_missing_features():
    data = {}

    response = requests.post(
        BASE_URL + "/predict",
        json=data
    )

    print("\n--- ERROR TEST ---")
    print("Status Code:", response.status_code)
    print("Response:", response.json())


if __name__ == "__main__":

    test_home()
    test_health()
    test_prediction()
    test_missing_features()