"""
GitHub JWT Generator for Knowledge Graph Agent.

This module provides functionality to generate JSON Web Tokens (JWT)
for GitHub App authentication.
"""

import time
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GitHubJWTGenerator:
    """
    Generate JWT tokens for GitHub App authentication.
    
    This class handles:
    - Loading private keys
    - Generating JWT tokens
    - Token validation and expiration
    - GitHub App authentication
    """
    
    def __init__(
        self,
        app_id: str,
        private_key_path: Optional[str] = None,
        private_key_content: Optional[str] = None
    ):
        """
        Initialize the JWT generator.
        
        Args:
            app_id: GitHub App ID
            private_key_path: Path to private key file (.pem)
            private_key_content: Private key content as string
        """
        self.app_id = app_id
        self.private_key = self._load_private_key(private_key_path, private_key_content)
        
        if not self.private_key:
            raise ValueError("Private key is required for JWT generation")
        
        logger.info(f"Initialized GitHub JWT generator for App ID: {app_id}")
    
    def _load_private_key(self, key_path: Optional[str], key_content: Optional[str]) -> Optional[str]:
        """
        Load private key from file or content.
        
        Args:
            key_path: Path to private key file
            key_content: Private key content
            
        Returns:
            Private key as string
        """
        if key_content:
            return key_content
        
        if key_path:
            try:
                with open(key_path, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                logger.error(f"Private key file not found: {key_path}")
                return None
            except Exception as e:
                logger.error(f"Error reading private key file: {str(e)}")
                return None
        
        return None
    
    def generate_jwt(
        self,
        expiration_minutes: int = 10,
        issued_at: Optional[datetime] = None,
        not_before: Optional[datetime] = None
    ) -> str:
        """
        Generate a JWT token for GitHub App authentication.
        
        Args:
            expiration_minutes: Token expiration time in minutes
            issued_at: Token issued at time (default: now)
            not_before: Token not valid before time (default: now)
            
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        issued_at = issued_at or now
        not_before = not_before or now
        expiration = issued_at + timedelta(minutes=expiration_minutes)
        
        # JWT payload for GitHub Apps
        payload = {
            "iat": int(issued_at.timestamp()),
            "exp": int(expiration.timestamp()),
            "nbf": int(not_before.timestamp()),
            "iss": self.app_id
        }
        
        try:
            # Generate JWT using RS256 algorithm
            token = jwt.encode(
                payload,
                self.private_key,
                algorithm="RS256"
            )
            
            logger.info(f"Generated JWT token for App ID {self.app_id}, expires at {expiration}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating JWT: {str(e)}")
            raise
    
    def generate_installation_token(
        self,
        installation_id: str,
        expiration_minutes: int = 60
    ) -> str:
        """
        Generate an installation access token.
        
        This requires first generating a JWT, then using it to get an installation token.
        
        Args:
            installation_id: GitHub App installation ID
            expiration_minutes: Token expiration time
            
        Returns:
            Installation access token
        """
        # First generate JWT
        jwt_token = self.generate_jwt(expiration_minutes=expiration_minutes)
        
        # In a real implementation, you would use this JWT to make an API call
        # to GitHub's installation token endpoint
        # For now, we'll return the JWT as a placeholder
        
        logger.info(f"Generated installation token for installation {installation_id}")
        return jwt_token
    
    def validate_jwt(self, token: str) -> Dict[str, Any]:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            Decoded token payload
        """
        try:
            # Decode and verify the token
            payload = jwt.decode(
                token,
                self.private_key,
                algorithms=["RS256"],
                options={"verify_signature": True}
            )
            
            logger.info("JWT token validation successful")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.error("JWT token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid JWT token: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error validating JWT: {str(e)}")
            raise
    
    def get_token_info(self, token: str) -> Dict[str, Any]:
        """
        Get information about a JWT token without validating it.
        
        Args:
            token: JWT token
            
        Returns:
            Token information
        """
        try:
            # Decode without verification to get payload
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            
            # Calculate remaining time
            now = int(time.time())
            exp = payload.get("exp", 0)
            remaining_seconds = exp - now
            
            return {
                "app_id": payload.get("iss"),
                "issued_at": payload.get("iat"),
                "expires_at": exp,
                "remaining_seconds": max(0, remaining_seconds),
                "is_expired": remaining_seconds <= 0
            }
            
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            raise


def create_github_jwt_generator(
    app_id: str,
    private_key_path: Optional[str] = None,
    private_key_content: Optional[str] = None
) -> GitHubJWTGenerator:
    """
    Factory function to create a GitHub JWT generator.
    
    Args:
        app_id: GitHub App ID
        private_key_path: Path to private key file
        private_key_content: Private key content
        
    Returns:
        GitHubJWTGenerator instance
    """
    return GitHubJWTGenerator(
        app_id=app_id,
        private_key_path=private_key_path,
        private_key_content=private_key_content
    )


# Example usage and testing
def demo_jwt_generation():
    """Demo JWT generation with sample data."""
    print("🔐 GitHub JWT Generator Demo")
    print("=" * 40)
    
    # Sample data (replace with your actual values)
    app_id = "123456"  # Your GitHub App ID
    private_key_content = """
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----
"""
    
    try:
        # Create JWT generator
        jwt_gen = create_github_jwt_generator(
            app_id=app_id,
            private_key_content=private_key_content
        )
        
        # Generate JWT
        token = jwt_gen.generate_jwt(expiration_minutes=10)
        print(f"✅ JWT generated: {token[:50]}...")
        
        # Get token info
        info = jwt_gen.get_token_info(token)
        print(f"✅ Token info: {info}")
        
        # Validate token
        payload = jwt_gen.validate_jwt(token)
        print(f"✅ Token validated: {payload}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False


if __name__ == "__main__":
    demo_jwt_generation() 