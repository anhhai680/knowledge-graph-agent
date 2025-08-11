"""
Data Model Analyzer for Generic Q&A Agent.

This module analyzes data models and database schemas in project repositories
following the EventFlowAnalyzer pattern detection methodology.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

from src.utils.logging import get_logger
from src.utils.defensive_programming import safe_len, ensure_list


class DatabaseType(str, Enum):
    """Database type enumeration."""
    RELATIONAL = "relational"
    NOSQL = "nosql"
    GRAPH = "graph"
    VECTOR = "vector"
    CACHE = "cache"
    UNKNOWN = "unknown"


@dataclass
class DataModel:
    """Data model information."""
    name: str
    fields: List[str]
    relationships: List[str]
    constraints: List[str]
    indexes: List[str]


@dataclass
class DataAnalysisResult:
    """Data model analysis results."""
    database_type: DatabaseType
    models: List[DataModel]
    relationships: List[str]
    migration_system: bool
    orm_framework: Optional[str]
    confidence_score: float


class DataModelAnalyzer:
    """Data model analyzer using EventFlowAnalyzer patterns."""

    def __init__(self):
        """Initialize data model analyzer."""
        self.logger = get_logger(self.__class__.__name__)
        self._db_patterns = self._initialize_db_patterns()
        self._orm_patterns = self._initialize_orm_patterns()

    def _initialize_db_patterns(self) -> Dict[DatabaseType, List[str]]:
        """Initialize database detection patterns."""
        return {
            DatabaseType.RELATIONAL: ["sqlite", "postgresql", "mysql", "sql"],
            DatabaseType.NOSQL: ["mongodb", "dynamodb", "cassandra", "nosql"],
            DatabaseType.GRAPH: ["neo4j", "memgraph", "graph"],
            DatabaseType.VECTOR: ["chroma", "pinecone", "weaviate", "vector"],
            DatabaseType.CACHE: ["redis", "memcached", "cache"]
        }

    def _initialize_orm_patterns(self) -> Dict[str, List[str]]:
        """Initialize ORM detection patterns."""
        return {
            "sqlalchemy": ["sqlalchemy", "declarative_base", "sessionmaker"],
            "django_orm": ["django.db", "models.Model"],
            "peewee": ["peewee", "Model"],
            "tortoise": ["tortoise", "Model"],
            "pydantic": ["pydantic", "BaseModel"]
        }

    async def analyze_data_models(
        self,
        repository_path: str,
        repository_context: Optional[Dict[str, Any]] = None
    ) -> DataAnalysisResult:
        """Analyze data models in repository."""
        self.logger.info(f"Analyzing data models for: {repository_path}")
        
        try:
            # Detect database type
            db_type = self._detect_database_type(repository_path)
            
            # Detect ORM framework
            orm_framework = self._detect_orm_framework(repository_path)
            
            # Extract data models
            models = self._extract_data_models(repository_path, orm_framework)
            
            # Extract relationships
            relationships = self._extract_relationships(repository_path, models)
            
            # Check for migrations
            has_migrations = self._check_migration_system(repository_path)
            
            # Calculate confidence
            confidence = self._calculate_confidence(db_type, models, orm_framework)
            
            return DataAnalysisResult(
                database_type=db_type,
                models=models,
                relationships=relationships,
                migration_system=has_migrations,
                orm_framework=orm_framework,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Data model analysis failed: {e}")
            return self._create_fallback_result()

    def _detect_database_type(self, repository_path: str) -> DatabaseType:
        """Detect database type from repository."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            # Check requirements files
            for req_file in ["requirements.txt", "pyproject.toml", "Pipfile"]:
                req_path = repo_path / req_file
                if req_path.exists():
                    try:
                        content = req_path.read_text(encoding='utf-8', errors='ignore')
                        content_lower = content.lower()
                        
                        for db_type, patterns in self._db_patterns.items():
                            if any(pattern in content_lower for pattern in patterns):
                                return db_type
                    except Exception:
                        continue
            
            # Check source files
            for file_path in repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]
                    content_lower = content.lower()
                    
                    for db_type, patterns in self._db_patterns.items():
                        if any(pattern in content_lower for pattern in patterns):
                            return db_type
                except Exception:
                    continue
            
            return DatabaseType.UNKNOWN
            
        except Exception:
            return DatabaseType.UNKNOWN

    def _detect_orm_framework(self, repository_path: str) -> Optional[str]:
        """Detect ORM framework."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            for file_path in repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]
                    content_lower = content.lower()
                    
                    for orm, patterns in self._orm_patterns.items():
                        if any(pattern in content_lower for pattern in patterns):
                            return orm
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None

    def _extract_data_models(self, repository_path: str, orm_framework: Optional[str]) -> List[DataModel]:
        """Extract data models from repository."""
        models = []
        
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            # Look for model files
            model_files = []
            for pattern in ["**/models.py", "**/model.py", "**/entities.py"]:
                model_files.extend(repo_path.glob(pattern))
            
            for model_file in model_files:
                try:
                    content = model_file.read_text(encoding='utf-8', errors='ignore')
                    extracted_models = self._parse_models_from_content(content, orm_framework)
                    models.extend(extracted_models)
                except Exception:
                    continue
            
            # Add generic models if none found
            if not models:
                models.append(DataModel(
                    name="GenericModel",
                    fields=["id", "created_at", "updated_at"],
                    relationships=[],
                    constraints=["primary_key"],
                    indexes=["id_index"]
                ))
            
        except Exception:
            pass
        
        return models[:10]  # Limit to 10 models

    def _parse_models_from_content(self, content: str, orm_framework: Optional[str]) -> List[DataModel]:
        """Parse models from file content."""
        models = []
        lines = content.split('\n')
        
        current_model = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect class definitions
            if line_stripped.startswith('class ') and 'Model' in line:
                class_name = line_stripped.split()[1].split('(')[0]
                current_model = DataModel(
                    name=class_name,
                    fields=[],
                    relationships=[],
                    constraints=[],
                    indexes=[]
                )
                models.append(current_model)
            
            # Extract fields (simplified)
            elif current_model and ('=' in line_stripped and any(
                field_type in line_stripped.lower() 
                for field_type in ['field', 'column', 'str', 'int', 'bool', 'date']
            )):
                field_name = line_stripped.split('=')[0].strip()
                if field_name and not field_name.startswith('_'):
                    current_model.fields.append(field_name)
        
        return models

    def _extract_relationships(self, repository_path: str, models: List[DataModel]) -> List[str]:
        """Extract relationships between models."""
        relationships = []
        
        # Simple relationship detection
        for model in models:
            for field in model.fields:
                if any(rel_indicator in field.lower() 
                       for rel_indicator in ['foreign', 'relation', 'ref', '_id']):
                    relationships.append(f"{model.name} -> {field}")
        
        return relationships[:10]  # Limit relationships

    def _check_migration_system(self, repository_path: str) -> bool:
        """Check if migration system exists."""
        try:
            from pathlib import Path
            repo_path = Path(repository_path)
            
            # Look for migration directories
            migration_patterns = ["migrations", "alembic", "migrate"]
            
            for pattern in migration_patterns:
                if list(repo_path.glob(f"**/{pattern}")):
                    return True
            
            return False
            
        except Exception:
            return False

    def _calculate_confidence(
        self, 
        db_type: DatabaseType, 
        models: List[DataModel], 
        orm_framework: Optional[str]
    ) -> float:
        """Calculate analysis confidence."""
        base_confidence = 0.2
        
        # Boost for database type detection
        db_boost = 0.3 if db_type != DatabaseType.UNKNOWN else 0
        
        # Boost for models found
        model_boost = min(safe_len(models) * 0.1, 0.3)
        
        # Boost for ORM detection
        orm_boost = 0.2 if orm_framework else 0
        
        return min(base_confidence + db_boost + model_boost + orm_boost, 1.0)

    def _create_fallback_result(self) -> DataAnalysisResult:
        """Create fallback result when analysis fails."""
        return DataAnalysisResult(
            database_type=DatabaseType.UNKNOWN,
            models=[],
            relationships=[],
            migration_system=False,
            orm_framework=None,
            confidence_score=0.1
        )