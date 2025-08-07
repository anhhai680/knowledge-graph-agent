# [TASK012] - Query Intent Detection Bug Fix

**Status:** Completed  
**Added:** August 6, 2025  
**Updated:** August 6, 2025

## Original Request
Act as an expert Python developer, you have to analyze the rag_agent and query_workflow to find the root cause and fix this issue why query_intent is always `CODE_SEARCH` even the users tried to different kind of requests.

## Thought Process
The user reported that query_intent was always being set to `CODE_SEARCH` regardless of the actual user query. This suggested a systematic issue in the query intent detection and classification system. I needed to trace the flow from user input through the RAG agent, query workflow, and orchestrator to identify where the intent determination was failing.

## Root Cause Analysis

After detailed investigation, I discovered **four critical issues** causing this behavior:

### Issue 1: RAG Agent Hardcoded Fallback
**Location:** `src/agents/rag_agent.py:150`  
**Problem:** The RAG agent was pre-setting intent to `CODE_SEARCH` as a fallback instead of allowing the workflow to determine it:
```python
"intent": query_intent or QueryIntent.CODE_SEARCH,  # ❌ WRONG
```

### Issue 2: Incorrect Workflow Execution Method  
**Location:** `src/agents/rag_agent.py:165`  
**Problem:** The RAG agent was calling `_execute_workflow(query_data)` with a dictionary, bypassing the proper workflow execution that includes intent analysis.

### Issue 3: Incorrect Enum Value Extraction
**Location:** `src/api/routes.py:347`  
**Problem:** The API was using `str(enum)` which returns `"QueryIntent.CODE_SEARCH"` instead of the enum value `"code_search"`.

### Issue 4: Data Type Inconsistencies
**Problem:** The RAG agent expected Document objects but received dictionaries from QueryState, causing processing errors.

## Implementation Plan

### Phase 1: Fix RAG Agent Intent Handling ✅
- [x] Remove hardcoded `CODE_SEARCH` fallback in query data creation
- [x] Change from `_execute_workflow()` to proper `workflow.run()` method
- [x] Update data processing to handle QueryState structure
- [x] Add `_format_sources_from_dict()` method for dictionary format

### Phase 2: Fix API Enum Value Extraction ✅  
- [x] Replace `str(enum)` with proper `enum.value` extraction
- [x] Add null safety for query_intent extraction
- [x] Update mapping function calls

### Phase 3: Verify Intent Detection Logic ✅
- [x] Test query intent determination algorithm in isolation
- [x] Verify orchestrator step execution
- [x] Confirm workflow state management

## Progress Tracking

**Overall Status:** Completed - 100%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 1.1 | Fix RAG agent hardcoded intent fallback | Complete | 2025-08-06 | Removed `or QueryIntent.CODE_SEARCH` |
| 1.2 | Update workflow execution method | Complete | 2025-08-06 | Changed to `workflow.run()` |
| 1.3 | Fix data type inconsistencies | Complete | 2025-08-06 | Added dict-to-Document conversion |
| 1.4 | Fix API enum value extraction | Complete | 2025-08-06 | Use `.value` instead of `str()` |
| 1.5 | Test intent detection algorithm | Complete | 2025-08-06 | Verified logic works correctly |

## Progress Log

### August 6, 2025
- **Root Cause Identified**: Found 4 critical issues in intent detection pipeline
- **Issue 1 Fixed**: Removed hardcoded `CODE_SEARCH` fallback in RAG agent 
- **Issue 2 Fixed**: Updated RAG agent to use proper `workflow.run()` method
- **Issue 3 Fixed**: Fixed API routes to use `enum.value` instead of `str(enum)`
- **Issue 4 Fixed**: Added proper data type handling for QueryState
- **Testing Completed**: Verified intent detection works for all query types:
  - ✅ "explain how authentication works" → `EXPLANATION`
  - ✅ "fix the login bug" → `DEBUGGING`  
  - ✅ "show me the api documentation" → `DOCUMENTATION`
  - ✅ "show system architecture" → `ARCHITECTURE`
  - ✅ "find the getUserById function" → `CODE_SEARCH`
- **Task Completed**: All fixes implemented and tested successfully

## Technical Details

### Files Modified
1. **`src/agents/rag_agent.py`**
   - Removed hardcoded intent fallback
   - Changed to proper workflow execution method
   - Added data type conversion for QueryState compatibility
   - Added `_format_sources_from_dict()` method

2. **`src/api/routes.py`**  
   - Fixed enum value extraction using `.value`
   - Added null safety for intent extraction
   - Improved mapping function usage

### Key Learning
The issue was not in the intent detection algorithm itself (which works correctly), but in the execution flow where the RAG agent was bypassing the workflow steps that perform intent analysis.

## Impact
- ✅ Query intent is now correctly determined based on user query content
- ✅ Different query types (explanation, debugging, documentation, architecture, code search) are properly classified
- ✅ API responses now show accurate intent values
- ✅ RAG processing can now adapt based on detected intent
- ✅ System maintains full backward compatibility
