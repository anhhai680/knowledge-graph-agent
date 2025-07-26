#!/usr/bin/env python3
"""
Run All Tests for Knowledge Graph Agent.

This script runs all test suites for the Knowledge Graph Agent API.
"""

import subprocess
import sys
import time
import os
from typing import List, Dict, Any

def run_test(test_file: str) -> Dict[str, Any]:
    """Run a single test file and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"⚠️  Warnings/Errors: {result.stderr}")
        
        return {
            "file": test_file,
            "success": success,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ TIMEOUT: {test_file} took longer than 2 minutes")
        return {
            "file": test_file,
            "success": False,
            "duration": duration,
            "returncode": -1,
            "stdout": "",
            "stderr": "Timeout after 2 minutes"
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ ERROR running {test_file}: {e}")
        return {
            "file": test_file,
            "success": False,
            "duration": duration,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }

def run_health_check():
    """Check if the server is running."""
    print("🏥 Checking if server is running...")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print("❌ Server is not responding correctly")
            return False
    except Exception as e:
        print(f"❌ Server is not running: {e}")
        print("💡 Start the server with: python main.py")
        return False

def get_test_files() -> List[str]:
    """Get all test files in the test directory."""
    test_files = []
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file in os.listdir(test_dir):
        if file.endswith('.py') and file.startswith('test_') and file != '__init__.py':
            test_files.append(os.path.join(test_dir, file))
    
    # Sort test files for consistent order
    test_files.sort()
    return test_files

def print_summary(results: List[Dict[str, Any]]):
    """Print test summary."""
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result["success"])
    failed_tests = total_tests - passed_tests
    total_duration = sum(result["duration"] for result in results)
    
    print(f"Total Test Files: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Total Duration: {total_duration:.2f}s")
    
    if failed_tests > 0:
        print(f"\n❌ Failed Tests:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['file']} (Duration: {result['duration']:.2f}s)")
                if result["stderr"]:
                    print(f"    Error: {result['stderr'][:100]}...")
    
    print(f"\n✅ Passed Tests:")
    for result in results:
        if result["success"]:
            print(f"  - {result['file']} (Duration: {result['duration']:.2f}s)")
    
    print(f"\n💡 Recommendations:")
    if failed_tests == 0:
        print("   🎉 All tests passed! The Knowledge Graph Agent is working perfectly.")
    else:
        print("   🔧 Some tests failed. Check the error messages above.")
        print("   🚀 Make sure the server is running: python main.py")
        print("   📚 Check the API documentation: http://localhost:8000/docs")

def main():
    """Main function to run all tests."""
    print("🚀 Knowledge Graph Agent - Comprehensive Test Suite")
    print("=" * 80)
    
    # Check if server is running
    if not run_health_check():
        print("\n❌ Cannot run tests - server is not running!")
        print("💡 Please start the server first:")
        print("   1. Start ChromaDB: docker-compose up -d chroma")
        print("   2. Start server: python main.py")
        print("   3. Run tests: python test/run_all_tests.py")
        return
    
    # Get all test files
    test_files = get_test_files()
    
    if not test_files:
        print("❌ No test files found!")
        return
    
    print(f"\n📋 Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"   - {os.path.basename(test_file)}")
    
    # Run all tests
    results = []
    for test_file in test_files:
        result = run_test(test_file)
        results.append(result)
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    failed_tests = sum(1 for result in results if not result["success"])
    if failed_tests > 0:
        print(f"\n❌ {failed_tests} test(s) failed!")
        sys.exit(1)
    else:
        print(f"\n🎉 All {len(results)} tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main() 