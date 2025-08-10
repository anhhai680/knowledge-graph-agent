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
        
        # Add actors from workflow entities (only if they are meaningful business terms)
        meaningful_entities = ['user', 'customer', 'client', 'admin', 'service', 'system', 'order', 'payment', 'notification']
        for entity in workflow.entities:
            if any(meaningful in entity.lower() for meaningful in meaningful_entities):
                actor_candidates.add(entity.lower())
        
        # Add actors from code references (improved logic for car marketplace services)
        for ref in code_refs:
            # Extract actors from repository names (car marketplace specific)
            if ref.repository and ref.repository != 'unknown':
                repo_name = ref.repository.split('/')[-1].lower()
                
                # Car marketplace specific mapping
                if 'order' in repo_name:
                    actor_candidates.add('orderservice')
                elif 'notification' in repo_name:
                    actor_candidates.add('notificationservice')
                elif 'listing' in repo_name:
                    actor_candidates.add('listingservice')
                elif 'web' in repo_name or 'client' in repo_name:
                    actor_candidates.add('webclient')
                elif 'auth' in repo_name or 'user' in repo_name:
                    actor_candidates.add('userservice')
                elif 'payment' in repo_name:
                    actor_candidates.add('paymentservice')
            
            # Extract actors from class names and method contexts
            if ref.method_name:
                method_lower = ref.method_name.lower()
                if 'order' in method_lower:
                    actor_candidates.add('orderservice')
                elif 'notification' in method_lower or 'notify' in method_lower:
                    actor_candidates.add('notificationservice')
                elif 'user' in method_lower or 'auth' in method_lower:
                    actor_candidates.add('userservice')
        
        # Always include a user actor for user-initiated workflows
        if workflow.workflow in [WorkflowPattern.ORDER_PROCESSING, WorkflowPattern.USER_AUTHENTICATION]:
            actor_candidates.add('user')
        
        # Ensure we have a web client for web-based workflows
        if workflow.workflow == WorkflowPattern.ORDER_PROCESSING:
            actor_candidates.add('webclient')
        
        # Convert to Actor objects with proper validation and specific ordering
        actors = []
        valid_actor_names = set()
        
        # Prioritize specific actors for car marketplace order flow
        priority_actors = ['user', 'webclient', 'orderservice', 'notificationservice', 'userservice', 'paymentservice']
        
        # Add priority actors first
        for priority in priority_actors:
            if priority in actor_candidates and priority not in valid_actor_names:
                actor_type = self._classify_actor_type(priority)
                display_name = self._format_actor_name(priority)
                
                if display_name and len(display_name) >= 3:
                    actors.append(Actor(
                        name=priority,
                        display_name=display_name,
                        actor_type=actor_type,
                        description=f"{display_name} component"
                    ))
                    valid_actor_names.add(priority)
        
        # Add remaining candidates up to max limit
        remaining_candidates = actor_candidates - valid_actor_names
        for candidate in list(remaining_candidates)[:self.max_actors - len(actors)]:
            # Ensure actor name is valid (no special characters, proper format)
            clean_candidate = re.sub(r'[^a-zA-Z0-9]', '', candidate).lower()
            if len(clean_candidate) < 3 or clean_candidate in valid_actor_names:
                continue
                
            actor_type = self._classify_actor_type(clean_candidate)
            display_name = self._format_actor_name(clean_candidate)
            
            # Ensure display name is also clean
            clean_display = re.sub(r'[^a-zA-Z0-9\s]', '', display_name).strip()
            if not clean_display:
                continue
            
            actor = Actor(
                name=clean_candidate,
                display_name=clean_display,
                actor_type=actor_type,
                description=f"{actor_type.title()} component"
            )
            actors.append(actor)
            valid_actor_names.add(clean_candidate)
        
        # Ensure we have at least a basic set of actors if nothing meaningful was found
        if len(actors) < 2:
            actors.extend(self._get_default_actors(workflow.workflow))
        
        return actors[:self.max_actors]
    
    def _generate_sequence_steps(self, workflow: EventFlowQuery, code_refs: List[CodeReference], actors: List[Actor]) -> List[SequenceStep]:
        """Generate sequence steps from workflow and code references using only declared actors."""
        steps = []
        actor_names = [actor.name for actor in actors]
        
        # Only generate steps if we have declared actors
        if len(actors) < 2:
            self.logger.warning("Not enough valid actors for sequence diagram")
            return steps
        
        # Validate all steps only use declared actors
        if workflow.workflow == WorkflowPattern.ORDER_PROCESSING:
            steps.extend(self._generate_order_processing_steps(actors, code_refs))
        elif workflow.workflow == WorkflowPattern.USER_AUTHENTICATION:
            steps.extend(self._generate_auth_steps(actors, code_refs))
        elif workflow.workflow == WorkflowPattern.API_REQUEST_FLOW:
            steps.extend(self._generate_api_flow_steps(actors, code_refs))
        else:
            steps.extend(self._generate_generic_steps(workflow, actors, code_refs))
        
        # Filter steps to ensure all participants are declared
        valid_steps = []
        for step in steps:
            if step.actor in actor_names and step.target in actor_names:
                valid_steps.append(step)
            else:
                self.logger.debug(f"Skipping step with undeclared participants: {step.actor} -> {step.target}")
        
        # Assign order and limit steps
        for i, step in enumerate(valid_steps):
            step.order = i + 1
        
        return valid_steps[:self.max_steps]
    
    def _generate_order_processing_steps(self, actors: List[Actor], code_refs: List[CodeReference]) -> List[SequenceStep]:
        """Generate steps for order processing workflow using only declared actors."""
        steps = []
        
        # Convert actors to a dictionary for easy lookup
        actor_dict = {actor.name: actor.display_name for actor in actors}
        actor_names = list(actor_dict.keys())
        
        # Find relevant actors from the declared actors list
        user_actor = None
        order_service = None
        notification_service = None
        web_client = None
        
        # Map declared actors to workflow roles (use only what we have)
        for actor in actors:
            actor_name_lower = actor.name.lower()
            
            if 'user' in actor_name_lower:
                user_actor = actor.name
            elif 'order' in actor_name_lower:
                order_service = actor.name
            elif 'notification' in actor_name_lower:
                notification_service = actor.name
            elif 'client' in actor_name_lower or 'web' in actor_name_lower:
                web_client = actor.name
        
        # Use fallbacks from the declared actor list only
        if not user_actor and len(actors) > 0:
            user_actor = actors[0].name
        if not web_client and len(actors) > 1:
            web_client = actors[1].name
        if not order_service and len(actors) > 2:
            order_service = actors[2].name
        elif not order_service and web_client and web_client != user_actor:
            order_service = web_client  # Fallback to web client if no order service
        
        # Generate realistic car order flow steps using ONLY declared actors
        if user_actor and web_client:
            # User interacts with web client
            steps.append(SequenceStep(user_actor, web_client, "Access Car Marketplace", step_type="sync"))
            
            if order_service and order_service != web_client:
                # Web client communicates with order service
                steps.append(SequenceStep(web_client, order_service, "Submit Car Order", step_type="sync"))
                steps.append(SequenceStep(order_service, order_service, "Validate Order Details", step_type="sync"))
                steps.append(SequenceStep(order_service, order_service, "Process Payment", step_type="sync"))
                
                # Order service sends confirmation back to web client
                steps.append(SequenceStep(order_service, web_client, "Order Processed", step_type="sync"))
                
                # Notification service integration (if available)
                if notification_service:
                    steps.append(SequenceStep(order_service, notification_service, "Send Order Confirmation", step_type="async"))
                    steps.append(SequenceStep(notification_service, user_actor, "Order Confirmation Email/SMS", step_type="async"))
                
                # Web client notifies user
                steps.append(SequenceStep(web_client, user_actor, "Display Order Success", step_type="sync"))
            else:
                # Simplified flow when web client handles everything
                steps.append(SequenceStep(user_actor, web_client, "Submit Car Order", step_type="sync"))
                steps.append(SequenceStep(web_client, web_client, "Process Order", step_type="sync"))
                
                # Notification service integration (if available)
                if notification_service:
                    steps.append(SequenceStep(web_client, notification_service, "Send Confirmation", step_type="async"))
                    steps.append(SequenceStep(notification_service, user_actor, "Order Confirmation", step_type="async"))
                
                steps.append(SequenceStep(web_client, user_actor, "Order Confirmation", step_type="sync"))
        
        # Try to map code references to steps
        self._map_code_references_to_steps(steps, code_refs)
        
        return steps
    
    def _generate_auth_steps(self, actors: List[Actor], code_refs: List[CodeReference]) -> List[SequenceStep]:
        """Generate steps for authentication workflow."""
        steps = []
        
        user_actor = next((a.name for a in actors if 'user' in a.name.lower()), actors[0].name if actors else "user")
        auth_actor = next((a.name for a in actors if 'auth' in a.name.lower() or 'service' in a.name.lower()), actors[-1].name if actors else "service")
        
        steps.append(SequenceStep(user_actor, auth_actor, "Login Request", step_type="sync"))
        steps.append(SequenceStep(auth_actor, auth_actor, "Validate Credentials", step_type="sync"))
        steps.append(SequenceStep(auth_actor, user_actor, "Authentication Success", step_type="sync"))
        
        self._map_code_references_to_steps(steps, code_refs)
        return steps
    
    def _generate_api_flow_steps(self, actors: List[Actor], code_refs: List[CodeReference]) -> List[SequenceStep]:
        """Generate steps for API request flow."""
        steps = []
        
        client_actor = next((a.name for a in actors if 'client' in a.name.lower() or 'user' in a.name.lower()), actors[0].name if actors else "client")
        api_actor = next((a.name for a in actors if 'api' in a.name.lower() or 'service' in a.name.lower()), actors[-1].name if actors else "service")
        
        steps.append(SequenceStep(client_actor, api_actor, "API Request", step_type="sync"))
        steps.append(SequenceStep(api_actor, api_actor, "Process Request", step_type="sync"))
        steps.append(SequenceStep(api_actor, client_actor, "API Response", step_type="sync"))
        
        self._map_code_references_to_steps(steps, code_refs)
        return steps
    
    def _generate_generic_steps(self, workflow: EventFlowQuery, actors: List[Actor], code_refs: List[CodeReference]) -> List[SequenceStep]:
        """Generate generic steps based on workflow actions."""
        steps = []
        
        if len(actors) >= 2:
            source_actor = actors[0].name
            target_actor = actors[1].name
            
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
        # Special mappings for car marketplace services
        special_mappings = {
            'orderservice': 'Order Service',
            'notificationservice': 'Notification Service',
            'webclient': 'Web Client',
            'userservice': 'User Service',
            'paymentservice': 'Payment Service',
            'listingservice': 'Listing Service',
            'user': 'User'
        }
        
        # Check for exact matches first
        name_lower = name.lower()
        if name_lower in special_mappings:
            return special_mappings[name_lower]
        
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