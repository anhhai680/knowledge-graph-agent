"""
LLM-Enhanced Analyzer for Generic Q&A Agent.

This module provides LLM-powered analysis to generate intelligent insights
from the extracted repository data, providing contextual explanations and
architectural recommendations.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from src.llm.llm_factory import LLMFactory
from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len


class LLMEnhancedAnalyzer:
    """
    LLM-powered analyzer for providing intelligent insights on repository data.
    
    This analyzer takes the extracted code data and uses LLM to generate
    contextual explanations, architectural insights, and recommendations.
    """

    def __init__(self):
        """Initialize the LLM-enhanced analyzer."""
        self.logger = get_logger(self.__class__.__name__)
        self.settings = get_settings()
        self.llm = None  # Lazy initialization

    def _get_llm(self):
        """Get LLM instance with lazy initialization."""
        if self.llm is None:
            self.llm = LLMFactory.create()
        return self.llm

    async def enhance_api_analysis(
        self, 
        analysis_results: Dict[str, Any], 
        question: str,
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance API analysis with LLM-powered insights.
        
        Args:
            analysis_results: Raw analysis results from code extraction
            question: Original user question
            repository_context: Repository context information
            
        Returns:
            Enhanced analysis with LLM insights
        """
        try:
            endpoints = analysis_results.get("endpoints", [])
            frameworks = analysis_results.get("frameworks_detected", [])
            repository = analysis_results.get("repository", "unknown")
            
            if not endpoints:
                return analysis_results
            
            # Create prompt for LLM analysis
            prompt = self._create_api_analysis_prompt(
                endpoints, frameworks, repository, question
            )
            
            # Get LLM insights
            llm = self._get_llm()
            llm_response = await llm.ainvoke(prompt)
            
            # Extract text content from LLM response
            if hasattr(llm_response, 'content'):
                insights_text = str(llm_response.content)
            else:
                insights_text = str(llm_response)
            
            # Parse LLM response and enhance analysis
            enhanced_analysis = analysis_results.copy()
            enhanced_analysis["llm_insights"] = {
                "architectural_assessment": self._extract_architectural_insights(insights_text),
                "business_context": self._extract_business_insights(insights_text),
                "usage_patterns": self._extract_usage_patterns(insights_text),
                "recommendations": self._extract_recommendations(insights_text),
                "generated_at": datetime.now().isoformat()
            }
            
            # Add enhanced endpoint descriptions
            enhanced_analysis["enhanced_endpoints"] = self._enhance_endpoint_descriptions(
                endpoints, insights_text
            )
            
            self.logger.info(f"Enhanced API analysis with LLM insights for {repository}")
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"LLM enhancement failed: {e}")
            # Return original analysis if LLM fails
            return analysis_results

    def _create_api_analysis_prompt(
        self, 
        endpoints: List[Dict], 
        frameworks: List[str], 
        repository: str,
        question: str
    ) -> str:
        """Create prompt for LLM API analysis."""
        
        endpoints_text = "\n".join([
            f"- {ep.get('method', 'UNKNOWN')} {ep.get('path', '')}: {ep.get('description', 'No description')}"
            for ep in endpoints[:10]  # Limit to first 10 endpoints
        ])
        
        frameworks_text = ", ".join(frameworks) if frameworks else "Not detected"
        
        prompt = f"""
As a senior software architect, analyze the following API endpoints from the {repository} repository and provide insights:

**Repository**: {repository}
**Question**: {question}
**Frameworks Detected**: {frameworks_text}

**API Endpoints**:
{endpoints_text}

Please provide a comprehensive analysis including:

1. **Architectural Assessment**: What architectural patterns and design principles are evident?
2. **Business Context**: What business domain and capabilities does this API serve?
3. **Usage Patterns**: How would these endpoints typically be used together?
4. **Technical Recommendations**: Any suggestions for improvement or best practices?

Focus on providing actionable insights that would be valuable to developers working with this API.
Respond in a structured format with clear sections.
"""
        
        return prompt

    def _extract_architectural_insights(self, llm_response: str) -> str:
        """Extract architectural insights from LLM response."""
        try:
            # Look for architectural assessment section
            lines = llm_response.split('\n')
            architectural_section = []
            in_arch_section = False
            
            for line in lines:
                if 'architectural' in line.lower() and ('assessment' in line.lower() or 'analysis' in line.lower()):
                    in_arch_section = True
                    continue
                elif in_arch_section and line.strip().startswith(('2.', '**2.', '**Business')):
                    break
                elif in_arch_section and line.strip():
                    architectural_section.append(line.strip())
            
            return ' '.join(architectural_section) if architectural_section else "Standard REST API architecture detected"
            
        except Exception as e:
            self.logger.warning(f"Error extracting architectural insights: {e}")
            return "REST API with CRUD operations following standard patterns"

    def _extract_business_insights(self, llm_response: str) -> str:
        """Extract business context insights from LLM response."""
        try:
            lines = llm_response.split('\n')
            business_section = []
            in_business_section = False
            
            for line in lines:
                if 'business' in line.lower() and ('context' in line.lower() or 'domain' in line.lower()):
                    in_business_section = True
                    continue
                elif in_business_section and line.strip().startswith(('3.', '**3.', '**Usage')):
                    break
                elif in_business_section and line.strip():
                    business_section.append(line.strip())
            
            return ' '.join(business_section) if business_section else "Core business functionality for data management"
            
        except Exception as e:
            self.logger.warning(f"Error extracting business insights: {e}")
            return "Provides essential business operations and data management capabilities"

    def _extract_usage_patterns(self, llm_response: str) -> str:
        """Extract usage patterns from LLM response."""
        try:
            lines = llm_response.split('\n')
            usage_section = []
            in_usage_section = False
            
            for line in lines:
                if 'usage' in line.lower() and ('pattern' in line.lower() or 'workflow' in line.lower()):
                    in_usage_section = True
                    continue
                elif in_usage_section and line.strip().startswith(('4.', '**4.', '**Technical', '**Recommend')):
                    break
                elif in_usage_section and line.strip():
                    usage_section.append(line.strip())
            
            return ' '.join(usage_section) if usage_section else "Standard CRUD workflow patterns"
            
        except Exception as e:
            self.logger.warning(f"Error extracting usage patterns: {e}")
            return "Typical create, read, update, delete operations with standard REST patterns"

    def _extract_recommendations(self, llm_response: str) -> str:
        """Extract recommendations from LLM response."""
        try:
            lines = llm_response.split('\n')
            recommendations_section = []
            in_recommendations_section = False
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['recommendation', 'suggest', 'improvement']):
                    in_recommendations_section = True
                    continue
                elif in_recommendations_section and line.strip():
                    recommendations_section.append(line.strip())
            
            return ' '.join(recommendations_section) if recommendations_section else "Continue following REST best practices"
            
        except Exception as e:
            self.logger.warning(f"Error extracting recommendations: {e}")
            return "Consider adding proper error handling, authentication, and API documentation"

    def _enhance_endpoint_descriptions(
        self, 
        endpoints: List[Dict], 
        llm_response: str
    ) -> List[Dict]:
        """Enhance endpoint descriptions with LLM insights."""
        enhanced = []
        
        for endpoint in endpoints:
            enhanced_endpoint = endpoint.copy()
            
            # Add enhanced description based on method and path
            method = endpoint.get("method", "")
            path = endpoint.get("path", "")
            
            if method == "GET" and "{id}" in path:
                enhanced_endpoint["usage_example"] = f"Retrieve specific item details by ID"
                enhanced_endpoint["typical_response"] = "Returns single item with full details"
            elif method == "GET":
                enhanced_endpoint["usage_example"] = f"List all items with optional filtering"
                enhanced_endpoint["typical_response"] = "Returns array of items"
            elif method == "POST":
                enhanced_endpoint["usage_example"] = f"Create new item with provided data"
                enhanced_endpoint["typical_response"] = "Returns created item with generated ID"
            elif method == "PUT":
                enhanced_endpoint["usage_example"] = f"Update entire item or create if not exists"
                enhanced_endpoint["typical_response"] = "Returns updated item data"
            elif method == "DELETE":
                enhanced_endpoint["usage_example"] = f"Remove item from system"
                enhanced_endpoint["typical_response"] = "Returns success confirmation"
            
            enhanced.append(enhanced_endpoint)
        
        return enhanced

    async def enhance_general_analysis(
        self,
        analysis_results: Dict[str, Any],
        question: str,
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance general analysis with LLM insights for non-API questions.
        
        Args:
            analysis_results: Raw analysis results
            question: Original user question
            repository_context: Repository context
            
        Returns:
            Enhanced analysis with LLM insights
        """
        try:
            repository = repository_context.get("repository", "unknown")
            frameworks = repository_context.get("frameworks", [])
            languages = repository_context.get("languages", [])
            document_count = repository_context.get("document_count", 0)
            
            # Create general analysis prompt
            prompt = f"""
As a senior software architect, analyze the following repository and answer the user's question:

**Repository**: {repository}
**Question**: {question}
**Languages**: {', '.join(languages) if languages else 'Not detected'}
**Frameworks**: {', '.join(frameworks) if frameworks else 'Not detected'}
**Documents Analyzed**: {document_count}

Based on the repository information, provide a comprehensive answer that includes:
1. Direct answer to the user's question
2. Relevant technical context
3. Architectural insights
4. Practical recommendations

Focus on being helpful and providing actionable information.
"""
            
            # Get LLM insights
            llm = self._get_llm()
            insights = await llm.ainvoke(prompt)
            
            # Enhance analysis with LLM insights
            enhanced_analysis = analysis_results.copy()
            enhanced_analysis["llm_insights"] = {
                "comprehensive_answer": insights,
                "generated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Enhanced general analysis with LLM insights for {repository}")
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"LLM enhancement failed for general analysis: {e}")
            return analysis_results
