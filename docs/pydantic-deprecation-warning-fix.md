# Pydantic Deprecation Warning Fix

## Problem

GitHub Actions was showing Pydantic deprecation warnings:

```
PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead
PydanticDeprecatedSince20: The `dict` method is deprecated; use `model_dump` instead
```

## Root Cause

The codebase was using deprecated Pydantic V1 patterns that are no longer recommended in Pydantic V2:

1. **Class-based Config**: Using `class Config:` instead of `model_config = ConfigDict()`
2. **Deprecated dict() method**: Using `.dict()` instead of `.model_dump()`

## Solution

Updated all deprecated Pydantic patterns to use the new V2 syntax.

### Changes Made

#### 1. **Class-based Config → ConfigDict**

**Before:**
```python
class WorkflowStateMetadata(BaseModel):
    # ... fields ...
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
```

**After:**
```python
class WorkflowStateMetadata(BaseModel):
    # ... fields ...
    
    model_config = ConfigDict(use_enum_values=True)
```

**Before:**
```python
class GitSettings(BaseModel):
    # ... fields ...
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "GIT_"
        case_sensitive = False
```

**After:**
```python
class GitSettings(BaseModel):
    # ... fields ...
    
    model_config = ConfigDict(env_prefix="GIT_", case_sensitive=False)
```

#### 2. **dict() method → model_dump()**

**Before:**
```python
config_dict = config.dict()
json.dump(metadata.dict(), f, indent=2)
```

**After:**
```python
config_dict = config.model_dump()
json.dump(metadata.model_dump(), f, indent=2)
```

#### 3. **Added ConfigDict Import**

Added `ConfigDict` to the imports in affected files:

```python
from pydantic import BaseModel, Field, ConfigDict
```

## Files Modified

1. **`src/workflows/state_manager.py`**
   - Updated `WorkflowStateMetadata` class config
   - Updated `.dict()` to `.model_dump()`
   - Added `ConfigDict` import

2. **`src/config/git_settings.py`**
   - Updated `GitSettings` class config
   - Added `ConfigDict` import

3. **`tests/unit/test_query_patterns_config_new.py`**
   - Updated `.dict()` to `.model_dump()`

## Verification

### Before Fix
```bash
python3 -m pytest tests/unit/test_query_patterns_config_new.py tests/unit/test_workflows.py -v
# Output: Multiple PydanticDeprecatedSince20 warnings
```

### After Fix
```bash
python3 -m pytest tests/unit/test_query_patterns_config_new.py tests/unit/test_workflows.py -v
# Output: 45 passed in 1.03s (no deprecation warnings)
```

## Pydantic V2 Migration Guide

### Config Changes
- **Old**: `class Config:`
- **New**: `model_config = ConfigDict()`

### Method Changes
- **Old**: `.dict()`
- **New**: `.model_dump()`
- **Old**: `.json()`
- **New**: `.model_dump_json()`
- **Old**: `.copy()`
- **New**: `.model_copy()`

### Validation Changes
- **Old**: `@validator`
- **New**: `@field_validator`

## Best Practices

1. **Use ConfigDict**: Always use `ConfigDict` instead of class-based config
2. **Use model_dump()**: Use `.model_dump()` instead of `.dict()`
3. **Import ConfigDict**: Add `ConfigDict` to pydantic imports when needed
4. **Check Pydantic docs**: Refer to the official migration guide for complex cases

## Impact

✅ **Pydantic deprecation warnings eliminated**  
✅ **All tests still pass**  
✅ **Future-proof code using Pydantic V2 patterns**  
✅ **Clean test output in GitHub Actions**  
✅ **No functional changes to behavior**  

## Related Issues

This is part of the broader Pydantic V2 migration. Other potential deprecation warnings to watch for:

- `@validator` decorators
- `.json()` method usage
- `.copy()` method usage
- Custom validators using old patterns

The fix ensures the codebase is using modern Pydantic V2 patterns and eliminates deprecation warnings in CI/CD pipelines. 