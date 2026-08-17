import json

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))
    
    # SYNTAX ERROR: Missing a closing parenthesis on the print statement below
    # and a missing colon at the end of the 'if' statement line
    message = "Hello from Lambda!"
    
    if "key1" in event
        print(f"Key1 value: {event['key1']}"
        
    return {
        'statusCode': 200,
        'body': json.dumps(message)
    }
