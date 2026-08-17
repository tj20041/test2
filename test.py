import json

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))

    # Fixed: Added missing colon at end of if-statement
    message = "Hello from Lambda!"

    if "key1" in event:
        # Fixed: Added missing closing parenthesis on print call
        print(f"Key1 value: {event['key1']}")

    return {
        'statusCode': 200,
        'body': json.dumps(message)
    }
