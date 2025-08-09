# Pytest Collection Warning Fix

## Problem

GitHub Actions was showing pytest collection warnings:

```
tests/unit/test_base_agent.py:14
  /home/runner/work/knowledge-graph-agent/knowledge-graph-agent/tests/unit/test_base_agent.py:14: PytestCollectionWarning: cannot collect test class 'TestAgent' because it has a __init__ constructor (from: tests/unit/test_base_agent.py)
    class TestAgent(BaseAgent):

tests/unit/test_workflows.py:151
  /home/runner/work/knowledge-graph-agent/knowledge-graph-agent/tests/unit/test_workflows.py:151: PytestCollectionWarning: cannot collect test class 'TestWorkflow' because it has a __init__ constructor (from: tests/unit/test_workflows.py)
    class TestWorkflow(BaseWorkflow):
```

## Root Cause

Pytest cannot collect test classes that have `__init__` constructors. The classes `TestAgent` and `TestWorkflow` were inheriting from `BaseAgent` and `BaseWorkflow` respectively, which have constructors, causing pytest to try to collect them as test classes.

## Solution

Rename the classes that inherit from base classes to not start with "Test" so pytest doesn't try to collect them as test classes.

### Changes Made

#### 1. `tests/unit/test_base_agent.py`

**Before:**
```python
class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing."""
    
    def __init__(self, agent_name="TestAgent", workflow=None):
        super().__init__(agent_name=agent_name, workflow=workflow)
```

**After:**
```python
class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""
    
    def __init__(self, agent_name="TestAgent", workflow=None):
        super().__init__(agent_name=agent_name, workflow=workflow)
```

**Updated all references:**
- `TestAgent(agent_name="TestAgent")` → `MockAgent(agent_name="TestAgent")`
- All test methods now use `MockAgent` instead of `TestAgent`

#### 2. `tests/unit/test_workflows.py`

**Before:**
```python
class TestWorkflow(BaseWorkflow):
    """Test implementation of BaseWorkflow for testing."""
    
    def __init__(self, fail_step: str = None, **kwargs):
        super().__init__(**kwargs)
        self.fail_step = fail_step
        self.steps_executed = []
```

**After:**
```python
class MockWorkflow(BaseWorkflow):
    """Mock implementation of BaseWorkflow for testing."""
    
    def __init__(self, fail_step: str = None, **kwargs):
        super().__init__(**kwargs)
        self.fail_step = fail_step
        self.steps_executed = []
```

**Updated all references:**
- `TestWorkflow()` → `MockWorkflow()`
- `TestWorkflow(fail_step="process")` → `MockWorkflow(fail_step="process")`
- All test methods now use `MockWorkflow` instead of `TestWorkflow`

## Files Modified

1. **`tests/unit/test_base_agent.py`**
   - Renamed `TestAgent` class to `MockAgent`
   - Updated all 15+ references to use `MockAgent`

2. **`tests/unit/test_workflows.py`**
   - Renamed `TestWorkflow` class to `MockWorkflow`
   - Updated all 5+ references to use `MockWorkflow`

## Verification

### Before Fix
```bash
python3 -m pytest tests/unit/test_base_agent.py tests/unit/test_workflows.py --collect-only
# Output: PytestCollectionWarning: cannot collect test class 'TestAgent' because it has a __init__ constructor
# Output: PytestCollectionWarning: cannot collect test class 'TestWorkflow' because it has a __init__ constructor
```

### After Fix
```bash
python3 -m pytest tests/unit/test_base_agent.py tests/unit/test_workflows.py --collect-only
# Output: 50 tests collected in 0.64s (no warnings)

python3 -m pytest tests/unit/ --collect-only
# Output: 210 tests collected in 0.88s (no warnings)

python3 -m pytest tests/unit/ -v --tb=short --maxfail=5
# Output: 210 passed, 8 warnings in 1.30s
```

## Best Practices

1. **Test class naming**: Classes that inherit from base classes with constructors should not start with "Test"
2. **Use descriptive names**: Use names like `MockAgent`, `FakeWorkflow`, `DummyProcessor` instead of `TestAgent`
3. **Keep test classes simple**: Test classes should not have complex constructors or inheritance
4. **Use fixtures**: For complex setup, use pytest fixtures instead of class inheritance

## Impact

✅ **Pytest collection warnings eliminated**  
✅ **All tests still pass**  
✅ **No functional changes to test logic**  
✅ **GitHub Actions will run without warnings**  

## Related Issues

This is a common pytest issue when:
- Test classes inherit from classes with constructors
- Classes start with "Test" and have `__init__` methods
- Complex inheritance hierarchies in test files

The fix ensures clean test collection and eliminates warnings in CI/CD pipelines. 