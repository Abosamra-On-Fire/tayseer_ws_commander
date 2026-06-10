#!/usr/bin/env python3
import os
import json
from tayseer_commander.llm_client import GroqLLMClient


def test_basic_planning():
    """Test 1: Simple object movement."""
    print("=" * 60)
    print("TEST 1: Move blue cube to shelf")
    print("=" * 60)
    
    api_key = os.getenv("Groq_API_KEY")
    if not api_key:
        print("ERROR: Set Groq_API_KEY environment variable")
        print("   export Groq_API_KEY='your-key-here'")
        return
    
    client = GroqLLMClient(api_key)

    #MOck
    world_state = {
        "blue_cube": {
            "position": [1.2, 3.4, 0.0],
            "frame_id": "map",
            "last_seen": "2026-06-06T15:10:00"
        },
        "shelf": {
            "position": [5.0, 2.0, 0.0],
            "frame_id": "map",
            "last_seen": "2026-06-06T15:10:00"
        },
        "red_ball": {
            "position": [0.5, 1.0, 0.0],
            "frame_id": "map",
            "last_seen": "2026-06-06T15:10:00"
        }
    }
    
    prompt = "Move the blue cube to the shelf"
    result = client.generate_plan(prompt, world_state)
    
    print(f"\nUser prompt: {prompt}")
    print(f"\nWorld state: {json.dumps(world_state, indent=2)}")
    print(f"\n{'='*60}")
    print(f"LLM Reasoning: {result.get('reasoning')}")
    print(f"\nGenerated Plan ({len(result.get('plan', []))} steps):")
    print(json.dumps(result.get('plan'), indent=2))
    
    #validating plan structure
    plan = result.get('plan', [])
    assert len(plan) > 0, "Plan should not be empty"
    assert plan[0]['action'] == 'navigate_to', "First step should be navigate"
    assert 'params' in plan[0], "Each step should have params"
    
    print("\n TEST 1 PASSED: Plan generated and structure is valid")


def test_multi_object():
    """Test 2: Complex multi-object task."""
    print("\n" + "=" * 60)
    print("TEST 2: Pick up red ball and place it near blue cube")
    print("=" * 60)
    
    api_key = os.getenv("Groq_API_KEY")
    client = GroqLLMClient(api_key)
    
    world_state = {
        "red_ball": {"position": [0.5, 1.0, 0.0], "frame_id": "map"},
        "blue_cube": {"position": [1.2, 3.4, 0.0], "frame_id": "map"},
        "table": {"position": [2.0, 2.0, 0.0], "frame_id": "map"}
    }
    
    prompt = "Pick up the red ball and place it next to the blue cube"
    result = client.generate_plan(prompt, world_state)
    
    print(f"\nUser prompt: {prompt}")
    print(f"\nLLM Reasoning: {result.get('reasoning')}")
    print(f"\nGenerated Plan:")
    print(json.dumps(result.get('plan'), indent=2))
    
    plan = result.get('plan', [])
    assert any(step['action'] == 'pick' for step in plan), "Should include pick action"
    assert any(step['action'] == 'place' for step in plan), "Should include place action"
    
    print("\n TEST 2 PASSED: Multi-object plan valid")


def test_replanning():
    """Test 3: Replan after failure."""
    print("\n" + "=" * 60)
    print("TEST 3: Replan after pick failure")
    print("=" * 60)
    
    api_key = os.getenv("Groq_API_KEY")
    client = GroqLLMClient(api_key)
    
    world_state = {
        "blue_cube": {"position": [1.2, 3.4, 0.0], "frame_id": "map"}
    }
    
    failed_action = {
        "action": "pick",
        "params": {"object_name": "blue_cube"}
    }
    
    result = client.replan(
        "Move the blue cube to the shelf",
        world_state,
        failed_action,
        "Object slipped from gripper"
    )
    
    print(f"\nReplan reasoning: {result.get('reasoning')}")
    print(f"\nNew plan:")
    print(json.dumps(result.get('plan'), indent=2))
    
    print("\n TEST 3 COMPLETE: Replan generated")


def test_edge_case():
    """Test 4: Unknown object."""
    print("\n" + "=" * 60)
    print("TEST 4: Ask for object that doesn't exist")
    print("=" * 60)
    
    api_key = os.getenv("Groq_API_KEY")
    client = GroqLLMClient(api_key)
    
    world_state = {
        "blue_cube": {"position": [1.2, 3.4, 0.0], "frame_id": "map"}
    }
    
    prompt = "Pick up the green pyramid"
    result = client.generate_plan(prompt, world_state)
    
    print(f"\nUser prompt: {prompt}")
    print(f"Available objects: blue_cube")
    print(f"\nLLM Reasoning: {result.get('reasoning')}")
    print(f"Plan: {json.dumps(result.get('plan'), indent=2)}")
    
    # Should either be empty or explain it can't find it
    print("\n TEST 4 COMPLETE: Check if LLM handled unknown object correctly")


if __name__ == '__main__':
    try:
        test_basic_planning()
        test_multi_object()
        test_replanning()
        test_edge_case()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETE")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()