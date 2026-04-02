import requests
import json

def test_optimize_endpoint():
    url = "http://127.0.0.1:8000/optimize"
    
    # Payload structured for a Multi-Grid Ship (e.g., Drake Caterpillar's front modules)
    payload = {
        "ship_grids": [
            [[4, 10, 2], "Cargo Module 1"],
            [[4, 6, 2], "Front Module"]
        ],
        "manifest": [
            {"scu_type": "8 SCU", "quantity": 1, "priority": 1},
            {"scu_type": "4 SCU", "quantity": 3, "priority": 2},
            {"scu_type": "1 SCU", "quantity": 5, "priority": 3}
        ]
    }
    
    try:
        print(f"Sending POST request to {url} with payload:")
        print(json.dumps(payload, indent=2))
        
        response = requests.post(url, json=payload, timeout=15)
        print(f"\nStatus Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200 and response.json().get('success'):
            print("\n✅ API Test Passed: Valid Multi-Grid GNN inference response received.")
        else:
            print("\n❌ API Test Failed: Response was not successful.")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is it running?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_optimize_endpoint()
