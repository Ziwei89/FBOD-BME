import json



if __name__ == "__main__":
    data = [{
                "name": "Alice",
                "age": 18,
                "hobbies": ["reading", "traveling", "swimming"]
            },
            {
                "name": "ziwei",
                "age": 34,
                "hobbies": ["reading", "traveling", "swimming"]
            }]
    json_data = json.dumps(data)
    with open('data.json', 'w') as f:
        f.write(json_data)