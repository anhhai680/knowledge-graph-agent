#!/usr/bin/env python3
"""
Test script to check the actual API response format
"""

import requests
import json

def test_api_response():
    url = "http://localhost:8000/api/v1/query"
    payload = {
        "query": "Show me the system architecture",
        "top_k": 5,
        "include_metadata": True,
        "search_strategy": "hybrid"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        print("Response structure:")
        print(f"- response_type: {data.get('response_type')}")
        print(f"- generated_response type: {type(data.get('generated_response'))}")
        
        if data.get('generated_response'):
            print("\nGenerated response content preview:")
            gen_response = data['generated_response']
            if isinstance(gen_response, str):
                print("It's a string, first 200 chars:")
                print(gen_response[:200])
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(gen_response)
                    print("\nSuccessfully parsed as JSON!")
                    print(f"Keys: {list(parsed.keys())}")
                    if 'answer' in parsed:
                        print("\nAnswer preview (first 200 chars):")
                        print(parsed['answer'][:200])
                except json.JSONDecodeError:
                    print("\nNot valid JSON")
            else:
                print(f"It's a {type(gen_response)}: {gen_response}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_response()
