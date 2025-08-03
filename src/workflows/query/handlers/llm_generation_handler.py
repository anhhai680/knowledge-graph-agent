"""
LLM generation handler extending BaseWorkflow.

This module implements LLM generation operations by extending the existing 
BaseWorkflow infrastructure and using LLMFactory directly.
"""

import time
from typing import List
from src.workflows.base_workflow import BaseWorkflow
from src.workflows.workflow_states import (
    QueryState, 
    QueryIntent,
    ProcessingStatus, 
    update_workflow_progress
)
from src.llm.llm_factory import LLMFactory


class LLMGenerationHandler(BaseWorkflow[QueryState]):
    """
    Handle LLM generation operations.
    
    Extends BaseWorkflow to leverage existing error handling, retry logic,
    and progress tracking while focusing on LLM generation concerns.
    """
    
    def __init__(self, **kwargs):
        """Initialize LLM generation handler."""
        super().__init__(workflow_id="llm-generation", **kwargs)
        # Use existing factory directly (no service wrapper)
        self._llm = None
        
    def define_steps(self) -> List[str]:
        """Define the LLM generation workflow steps."""
        return ["generate_prompt", "call_llm", "process_response"]
    
    def execute_step(self, step: str, state: QueryState) -> QueryState:
        """
        Execute a single LLM generation step.
        
        Args:
            step: Step name to execute
            state: Current query state
            
        Returns:
            Updated query state
        """
        if step == "generate_prompt":
            # Generate contextual prompt
            context = state["retrieval_config"].get("prepared_context", "")
            prompt = self._generate_contextual_prompt(
                state["processed_query"],
                context,
                state["query_intent"] or QueryIntent.CODE_SEARCH
            )
            # Store prompt for LLM call
            state["retrieval_config"]["llm_prompt"] = prompt
            
        elif step == "call_llm":
            # Call LLM for response generation
            llm = self._get_llm()
            prompt = state["retrieval_config"].get("llm_prompt", "")
            
            generation_start = time.time()
            response = llm.invoke(prompt)
            generation_time = time.time() - generation_start
            
            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Ensure response_text is always a string
            if isinstance(response_text, list):
                response_text = "\n".join(str(item) for item in response_text)
            elif not isinstance(response_text, str):
                response_text = str(response_text)
            
            # Update LLM generation state
            state["llm_generation"]["status"] = ProcessingStatus.COMPLETED
            state["llm_generation"]["generated_response"] = response_text
            state["llm_generation"]["generation_time"] = generation_time
            state["generation_time"] = generation_time
            
        elif step == "process_response":
            # Process and validate the generated response
            response = state["llm_generation"]["generated_response"]
            self.logger.info(f"Generated response: {len(response)} characters")
            
            # Update progress using existing helper function
            state = update_workflow_progress(state, 90.0, "llm_generation_complete")
            
        return state
    
    def validate_state(self, state: QueryState) -> bool:
        """Validate that the state contains required fields for LLM generation."""
        return bool(state.get("retrieval_config", {}).get("prepared_context"))
    
    def _get_llm(self):
        """Get or create LLM instance using existing factory."""
        if self._llm is None:
            self._llm = LLMFactory.create()
        return self._llm

    def _generate_contextual_prompt(
        self,
        query: str,
        context: str,
        query_intent: QueryIntent
    ) -> str:
        """
        Generate contextual prompt for LLM.

        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency.

        Args:
            query: User query
            context: Formatted context from retrieved documents
            query_intent: Analyzed query intent

        Returns:
            Complete prompt for LLM
        """
        system_prompt = self._generate_system_prompt(query_intent)

        prompt_template = f"""{system_prompt}

Context from codebase:
{context}

User Question: {query}

Please provide a comprehensive answer based on the provided context. Include:
1. Direct answer to the question
2. Relevant code examples from the context
3. Source file references where applicable
4. Additional insights or recommendations

Answer:"""

        return prompt_template

    def _generate_system_prompt(self, query_intent: QueryIntent) -> str:
        """
        Generate system prompt based on query intent.

        This method reuses the exact logic from the original QueryWorkflow
        to maintain consistency.

        Args:
            query_intent: Analyzed query intent

        Returns:
            System prompt string
        """
        base_prompt = """You are an expert software engineer and code analyst. You help developers understand codebases by providing accurate, detailed, and contextual information based on the provided code context."""

        intent_specific_prompts = {
            QueryIntent.CODE_SEARCH: """
Focus on:
- Providing specific code examples and implementations
- Explaining how the code works and its purpose
- Identifying relevant patterns and best practices
- Suggesting similar implementations if applicable
""",
            QueryIntent.DOCUMENTATION: """
Focus on:
- Explaining the purpose and functionality clearly
- Providing comprehensive documentation-style responses
- Including usage examples and API information
- Highlighting important configuration details
""",
            QueryIntent.EXPLANATION: """
Focus on:
- Breaking down complex concepts into understandable parts
- Providing step-by-step explanations
- Using analogies and examples where helpful
- Connecting the explanation to the specific codebase context
""",
            QueryIntent.DEBUGGING: """
Focus on:
- Identifying potential issues and their causes
- Suggesting specific debugging approaches
- Providing troubleshooting steps
- Recommending fixes and improvements
""",
            QueryIntent.ARCHITECTURE: """
Focus on:
- Describing system structure and component relationships
- Explaining design patterns and architectural decisions
- Providing high-level overviews and data flow
- Identifying key components and their interactions
"""
        }

        return base_prompt + intent_specific_prompts.get(query_intent, intent_specific_prompts[QueryIntent.CODE_SEARCH])
