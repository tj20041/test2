import json

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))

    # Fixed: Added missing colon at end of if statement
    message = "Hello from Lambda!"

    # Fixed: Added missing colon after 'if' condition
    # Fixed: Added missing closing parenthesis on print call
    if "key1" in event:
        print(f"Key1 value: {event['key1']}")

    return {
        'statusCode': 200,
        'body': json.dumps(message)
    }
