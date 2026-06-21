import os
import json
from google import genai
from google.genai import types

def lookup_order(order_id: str) -> dict:
    with open('orders.json', 'r') as f:
        orders = json.load(f)
    return orders.get(order_id, {"error": "Order not found"})

def calculate(a: float, b: float, operator: str) -> float:
    if operator == '+':
        return a + b
    elif operator == '-':
        return a - b
    elif operator == '*':
        return a * b
    elif operator == '/':
        return a / b
    return 0.0

def run_loop(client, messages, max_steps=5):
    tools = [lookup_order, calculate]
    
    for step in range(max_steps):
        print(f"Step {step + 1}")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                tools=tools
            )
        )
        
        messages.append(response.candidates[0].content)
        
        if response.function_calls:
            for function_call in response.function_calls:
                print(f"Tool Call: {function_call.name} with args {function_call.args}")
                if function_call.name == "lookup_order":
                    result = lookup_order(**function_call.args)
                elif function_call.name == "calculate":
                    result = calculate(**function_call.args)
                else:
                    result = {"error": "Unknown tool"}
                
                print(f"Tool Result: {result}")
                
                function_response_part = types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": result}
                )
                
                messages.append(
                    types.Content(
                        role="user",
                        parts=[function_response_part]
                    )
                )
        else:
            print(f"Model Answer: {response.text}")
            break
    else:
        print("couldn't finish in time")
    
    return messages

def main():
    client = genai.Client()
    
    messages = [
        types.Content(role="user", parts=[types.Part.from_text("What did order A1001 cost?")])
    ]
    
    print("Turn 1")
    messages = run_loop(client, messages)
    
    print("\nTurn 2")
    messages.append(
        types.Content(role="user", parts=[types.Part.from_text("And what about three of them?")])
    )
    messages = run_loop(client, messages)

if __name__ == "__main__":
    main()
