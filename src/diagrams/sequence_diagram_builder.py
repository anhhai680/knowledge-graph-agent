"""
Sequence diagram builder for event flow visualization.

This module generates Mermaid sequence diagrams from workflow analysis and code references,
providing visual representation of event flows with proper actor identification.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import re

from src.analyzers.event_flow_analyzer import EventFlowQuery, WorkflowPattern
from src.discovery.code_discovery_engine import CodeReference
from src.utils.logging import get_logger


@dataclass
class SequenceStep:
    """A single step in a sequence diagram."""
    
    actor: str
    target: str  
    action: str
    code_reference: Optional[CodeReference] = None
    order: int = 0
    step_type: str = "sync"  # 'sync', 'async', 'event'
    description: Optional[str] = None


@dataclass
class Actor:
    """An actor in the sequence diagram."""
    
    name: str
    display_name: str
    actor_type: str  # 'user', 'service', 'system', 'external'
    description: Optional[str] = None


class SequenceDiagramBuilder:
    """
    Builder for Mermaid sequence diagrams from event flow analysis.
    
    This class creates sequence diagrams by analyzing workflow patterns
    and mapping code references to sequence steps and actors.
    """
    
    def __init__(self, max_actors: int = 10, max_steps: int = 20):
        """
        Initialize sequence diagram builder.
        
        Args:
            max_actors: Maximum number of actors in diagram
            max_steps: Maximum number of sequence steps
        """
        self.max_actors = max_actors
        self.max_steps = max_steps
        self.logger = get_logger(self.__class__.__name__)
        
        # Define actor type mappings
        self._actor_type_keywords = {
            'user': ['user', 'customer', 'client', 'person', 'admin'],
            'service': ['service', 'api', 'handler', 'processor', 'manager'],
            'system': ['system', 'database', 'storage', 'cache', 'queue'],
            'external': ['external', 'gateway', 'provider', 'webhook', 'third-party']
        }
        
        # Define step type patterns
        self._step_type_patterns = {
            'async': ['async', 'queue', 'message', 'event', 'notification', 'background'],
            'event': ['event', 'emit', 'publish', 'trigger', 'dispatch', 'notify']
        }
    
    def build_from_workflow(self, workflow: EventFlowQuery, code_refs: List[CodeReference]) -> str:
        """
        Build Mermaid sequence diagram from workflow and code references.
        
        Args:
            workflow: Parsed event flow query
            code_refs: List of relevant code references
            
        Returns:
            str: Mermaid sequence diagram syntax
        """
        try:
            # Discover actors from workflow and code references
            actors = self._discover_actors(workflow, code_refs)
            
            # Generate sequence steps
            steps = self._generate_sequence_steps(workflow, code_refs, actors)
            
            # Build Mermaid diagram
            diagram = self._generate_mermaid_syntax(actors, steps)
            
            # Optimize for readability
            optimized_diagram = self._optimize_diagram_layout(diagram)
            
            self.logger.info(f"Generated sequence diagram with {len(actors)} actors and {len(steps)} steps")
            return optimized_diagram
            
        except Exception as e:
            self.logger.error(f"Error building sequence diagram: {e}")
            # Return a fallback diagram
            return self._create_fallback_diagram(workflow)
    
    def _discover_actors(self, workflow: EventFlowQuery, code_refs: List[CodeReference]) -> List[Actor]:
        """Discover actors from workflow and code references."""
        actor_candidates = set()
        
        # Add actors from workflow entities
        for entity in workflow.entities:
            actor_candidates.add(entity)
        
        # Add actors from code references
        for ref in code_refs:
            # Extract potential actors from repository names
            repo_parts = ref.repository.split('/')[-1].split('-')
            for part in repo_parts:
                if len(part) > 3:  # Filter short words
                    actor_candidates.add(part)
            
            # Extract actors from file paths
            path_parts = ref.file_path.split('/')
            for part in path_parts:
                if 'service' in part.lower() or 'controller' in part.lower():
                    clean_name = re.sub(r'[^a-zA-Z]', '', part)
                    if len(clean_name) > 3:
                        actor_candidates.add(clean_name)
        
        # Always include a user actor for user-initiated workflows
        if workflow.workflow in [WorkflowPattern.ORDER_PROCESSING, WorkflowPattern.USER_AUTHENTICATION]:
            actor_candidates.add('user')
        
        # Convert to Actor objects
        actors = []
        for candidate in list(actor_candidates)[:self.max_actors]:
            actor_type = self._classify_actor_type(candidate)
            display_name = self._format_actor_name(candidate)
            
            actor = Actor(
                name=candidate,
                display_name=display_name,
                actor_type=actor_type,
                description=f"{actor_type.title()} component"
            )
            actors.append(actor)
        
        # Ensure we have at least a basic set of actors
        if len(actors) < 2:
            actors.extend(self._get_default_actors(workflow.workflow))
        
        return actors[:self.max_actors]
    
    def _generate_sequence_steps(self, workflow: EventFlowQuery, code_refs: List[CodeReference], actors: List[Actor]) -> List[SequenceStep]:
        """Generate sequence steps from workflow and code references."""
        steps = []
        actor_names = [actor.name for actor in actors]
        
        # Generate steps based on workflow pattern
        if workflow.workflow == WorkflowPattern.ORDER_PROCESSING:
            steps.extend(self._generate_order_processing_steps(actor_names, code_refs))
        elif workflow.workflow == WorkflowPattern.USER_AUTHENTICATION:
            steps.extend(self._generate_auth_steps(actor_names, code_refs))
        elif workflow.workflow == WorkflowPattern.API_REQUEST_FLOW:
            steps.extend(self._generate_api_flow_steps(actor_names, code_refs))
        else:
            steps.extend(self._generate_generic_steps(workflow, actor_names, code_refs))
        
        # Assign order and limit steps
        for i, step in enumerate(steps):
            step.order = i + 1
        
        return steps[:self.max_steps]
    
    def _generate_order_processing_steps(self, actors: List[str], code_refs: List[CodeReference]) -> List[SequenceStep]:
        """Generate steps for order processing workflow."""
        steps = []
        
        # Find relevant actors based on actual code references and repository names
        user_actor = next((a for a in actors if 'user' in a.lower() or 'client' in a.lower()), "User")
        
        # Extract service actors from actual repositories
        order_service = next((a for a in actors if 'order' in a.lower() and 'service' in a.lower()), None)
        if not order_service:
            order_service = next((a for a in actors if 'order' in a.lower()), "Order Service")
        
        payment_service = next((a for a in actors if 'payment' in a.lower()), "Payment Gateway")
        notification_service = next((a for a in actors if 'notification' in a.lower()), "Notification Service")
        listing_service = next((a for a in actors if 'listing' in a.lower() or 'catalog' in a.lower()), "Listing Service")
        
        # Generate realistic order flow steps based on actual code patterns
        steps.append(SequenceStep(user_actor, "Web Client", "Submit Order", step_type="sync"))
        steps.append(SequenceStep("Web Client", "API Gateway", "POST /api/orders", step_type="sync"))
        steps.append(SequenceStep("API Gateway", order_service, "Create Order Request", step_type="sync"))
        
        if listing_service != "Listing Service":  # If we found a real listing service
            steps.append(SequenceStep(order_service, listing_service, "Verify Product Availability", step_type="sync"))
            steps.append(SequenceStep(listing_service, order_service, "Product Available Response", step_type="sync"))
        
        steps.append(SequenceStep(order_service, "Database", "Save Order (Status: Pending)", step_type="sync"))
        steps.append(SequenceStep(order_service, "Message Queue", "Publish order.created event", step_type="event"))
        steps.append(SequenceStep("Message Queue", notification_service, "Consume order.created", step_type="event"))
        steps.append(SequenceStep(notification_service, user_actor, "Email: Order Confirmation", step_type="async"))
        
        steps.append(SequenceStep(user_actor, "Web Client", "Initiate Payment", step_type="sync"))
        steps.append(SequenceStep("Web Client", order_service, "POST /api/orders/{id}/payment", step_type="sync"))
        steps.append(SequenceStep(order_service, payment_service, "Process Payment", step_type="sync"))
        steps.append(SequenceStep(payment_service, order_service, "Payment Success", step_type="sync"))
        
        steps.append(SequenceStep(order_service, "Database", "Update Order (Status: Paid)", step_type="sync"))
        steps.append(SequenceStep(order_service, "Message Queue", "Publish order.payment.completed", step_type="event"))
        steps.append(SequenceStep("Message Queue", notification_service, "Consume payment completed", step_type="event"))
        steps.append(SequenceStep(notification_service, user_actor, "Email: Payment Receipt", step_type="async"))
        
        if listing_service != "Listing Service":  # Update inventory
            steps.append(SequenceStep(order_service, listing_service, "Update Product Status (sold)", step_type="sync"))
            steps.append(SequenceStep(listing_service, "Message Queue", "Publish product.sold event", step_type="event"))
        
        steps.append(SequenceStep(notification_service, user_actor, "Email: Purchase Complete", step_type="async"))
        
        # Try to map code references to steps
        self._map_code_references_to_steps(steps, code_refs)
        
        return steps
    
    def _generate_auth_steps(self, actors: List[str], code_refs: List[CodeReference]) -> List[SequenceStep]:
        """Generate steps for authentication workflow."""
        steps = []
        
        user_actor = next((a for a in actors if 'user' in a.lower()), actors[0])
        auth_actor = next((a for a in actors if 'auth' in a.lower() or 'service' in a.lower()), actors[-1])
        
        steps.append(SequenceStep(user_actor, auth_actor, "Login Request", step_type="sync"))
        steps.append(SequenceStep(auth_actor, auth_actor, "Validate Credentials", step_type="sync"))
        steps.append(SequenceStep(auth_actor, user_actor, "Authentication Success", step_type="sync"))
        
        self._map_code_references_to_steps(steps, code_refs)
        return steps
    
    def _generate_api_flow_steps(self, actors: List[str], code_refs: List[CodeReference]) -> List[SequenceStep]:
        """Generate steps for API request flow."""
        steps = []
        
        client_actor = next((a for a in actors if 'client' in a.lower() or 'user' in a.lower()), actors[0])
        api_actor = next((a for a in actors if 'api' in a.lower() or 'service' in a.lower()), actors[-1])
        
        steps.append(SequenceStep(client_actor, api_actor, "API Request", step_type="sync"))
        steps.append(SequenceStep(api_actor, api_actor, "Process Request", step_type="sync"))
        steps.append(SequenceStep(api_actor, client_actor, "API Response", step_type="sync"))
        
        self._map_code_references_to_steps(steps, code_refs)
        return steps
    
    def _generate_generic_steps(self, workflow: EventFlowQuery, actors: List[str], code_refs: List[CodeReference]) -> List[SequenceStep]:
        """Generate generic steps based on workflow actions."""
        steps = []
        
        if len(actors) >= 2:
            source_actor = actors[0]
            target_actor = actors[1]
            
            # Create steps from workflow actions
            for i, action in enumerate(workflow.actions):
                if i % 2 == 0:
                    step = SequenceStep(source_actor, target_actor, action.title(), step_type="sync")
                else:
                    step = SequenceStep(target_actor, source_actor, f"Response to {action}", step_type="sync")
                steps.append(step)
        
        self._map_code_references_to_steps(steps, code_refs)
        return steps
    
    def _map_code_references_to_steps(self, steps: List[SequenceStep], code_refs: List[CodeReference]) -> None:
        """Map code references to sequence steps."""
        for step in steps:
            # Find the most relevant code reference for this step
            best_ref = None
            best_score = 0.0
            
            for ref in code_refs:
                score = self._calculate_step_reference_score(step, ref)
                if score > best_score:
                    best_score = score
                    best_ref = ref
            
            if best_ref and best_score > 0.3:  # Threshold for relevance
                step.code_reference = best_ref
                step.description = f"Implemented in {best_ref.method_name}"
    
    def _calculate_step_reference_score(self, step: SequenceStep, ref: CodeReference) -> float:
        """Calculate relevance score between a step and code reference."""
        score = 0.0
        
        # Check if action matches method name or content
        action_lower = step.action.lower()
        if action_lower in ref.method_name.lower():
            score += 0.5
        if action_lower in ref.content_snippet.lower():
            score += 0.3
        
        # Check actor relevance
        if step.actor.lower() in ref.file_path.lower():
            score += 0.2
        
        return score
    
    def _classify_actor_type(self, actor_name: str) -> str:
        """Classify actor type based on name."""
        actor_lower = actor_name.lower()
        
        for actor_type, keywords in self._actor_type_keywords.items():
            if any(keyword in actor_lower for keyword in keywords):
                return actor_type
        
        return 'service'  # Default type
    
    def _format_actor_name(self, name: str) -> str:
        """Format actor name for display."""
        # Remove common suffixes and format
        clean_name = re.sub(r'(service|controller|handler|manager)$', '', name, flags=re.IGNORECASE)
        return clean_name.title()
    
    def _get_default_actors(self, workflow: WorkflowPattern) -> List[Actor]:
        """Get default actors for a workflow pattern."""
        defaults = {
            WorkflowPattern.ORDER_PROCESSING: [
                Actor("user", "User", "user"),
                Actor("orderservice", "Order Service", "service")
            ],
            WorkflowPattern.USER_AUTHENTICATION: [
                Actor("user", "User", "user"),
                Actor("authservice", "Auth Service", "service")
            ],
            WorkflowPattern.API_REQUEST_FLOW: [
                Actor("client", "Client", "user"),
                Actor("apiservice", "API Service", "service")
            ]
        }
        
        return defaults.get(workflow, [
            Actor("system", "System", "system"),
            Actor("service", "Service", "service")
        ])
    
    def _generate_mermaid_syntax(self, actors: List[Actor], steps: List[SequenceStep]) -> str:
        """Generate Mermaid sequence diagram syntax."""
        lines = ["```mermaid", "sequenceDiagram"]
        
        # Add participant declarations
        for actor in actors:
            lines.append(f"    participant {actor.name} as {actor.display_name}")
        
        lines.append("")  # Empty line for readability
        
        # Add sequence steps
        for step in steps:
            arrow = self._get_arrow_type(step.step_type)
            lines.append(f"    {step.actor}{arrow}{step.target}: {step.action}")
            
            # Add note if there's a code reference
            if step.code_reference:
                lines.append(f"    Note over {step.target}: {step.code_reference.file_path}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _get_arrow_type(self, step_type: str) -> str:
        """Get Mermaid arrow type for step type."""
        if step_type == "async":
            return "-))"
        elif step_type == "event":
            return "-->"
        else:
            return "->>"
    
    def _optimize_diagram_layout(self, diagram: str) -> str:
        """Optimize diagram for readability."""
        # Simple optimization - could be enhanced
        lines = diagram.split('\n')
        
        # Remove duplicate participant declarations
        seen_participants = set()
        optimized_lines = []
        
        for line in lines:
            if line.strip().startswith('participant'):
                participant_name = line.split()[1]
                if participant_name not in seen_participants:
                    seen_participants.add(participant_name)
                    optimized_lines.append(line)
            else:
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
    
    def _create_fallback_diagram(self, workflow: EventFlowQuery) -> str:
        """Create a simple fallback diagram when generation fails."""
        return f"""```mermaid
sequenceDiagram
    participant User as User
    participant System as System
    
    User->>System: {workflow.intent or 'Request'}
    System->>System: Process Request
    System->>User: Response
```"""