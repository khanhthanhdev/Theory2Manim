#!/usr/bin/env python3
"""
Test script for OpenRouter integration
"""
import os
from dotenv import load_dotenv
from mllm_tools.openrouter import OpenRouterWrapper

# Load environment variables
load_dotenv()

def test_openrouter_basic():
    """Test basic OpenRouter functionality"""
    print("Testing OpenRouter integration...")
    
    # Check if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No OPENROUTER_API_KEY found in environment variables")
        print("Please set your OpenRouter API key in .env file:")
        print("OPENROUTER_API_KEY=your_api_key_here")
        return False
    
    try:        # Test with a simple text message
        wrapper = OpenRouterWrapper(
            model_name="openrouter/deepseek/deepseek-chat",
            temperature=0.7,
            print_cost=True,
            verbose=True
        )
        
        messages = [
            {"type": "text", "content": "Hello! Can you explain what 2+2 equals in one sentence?"}
        ]
        
        print("Sending test message to OpenRouter...")
        response = wrapper(messages)
        
        if response and not response.startswith("Error"):
            print(f"✅ OpenRouter test successful!")
            print(f"Response: {response}")
            return True
        else:
            print(f"❌ OpenRouter test failed: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenRouter: {e}")
        return False

def test_openrouter_models():
    """Test different OpenRouter models"""
    models_to_test = [
        "openrouter/deepseek/deepseek-r1",
        "openrouter/qwen/qwen3-235b-a22b",
        "openrouter/deepseek/deepseek-chat-v3-0324",

    ]
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No OPENROUTER_API_KEY found")
        return
    
    messages = [
        {"type": "text", "content": "What is the derivative of x^2? Answer in one sentence."}
    ]
    
    for model in models_to_test:
        print(f"\nTesting model: {model}")
        try:
            wrapper = OpenRouterWrapper(
                model_name=model,
                temperature=0.3,
                print_cost=True
            )
            
            response = wrapper(messages)
            if response and not response.startswith("Error"):
                print(f"✅ {model}: Success")
                print(f"   Response: {response[:100]}...")
            else:
                print(f"❌ {model}: Failed - {response}")
                
        except Exception as e:
            print(f"❌ {model}: Error - {e}")

if __name__ == "__main__":
    print("=== OpenRouter Integration Test ===")
    
    # Test basic functionality
    success = test_openrouter_basic()
    
    if success:
        print("\n=== Testing Different Models ===")
        test_openrouter_models()
    
    print("\n=== Test Complete ===")
