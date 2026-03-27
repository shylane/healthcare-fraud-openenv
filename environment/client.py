"""
HTTP Client for the Healthcare Claims Fraud Detection Environment.

Provides a client interface that can connect to the environment
running as an HTTP server (locally or in a Docker container).

Updated for LLM-native API (text generation).
"""

import subprocess
import time
from typing import Optional, Dict, Any, Union
import requests

from .models import (
    ClaimAction,
    ClaimObservation,
    ClaimState,
    StepResult,
)


class HealthClaimEnv:
    """
    HTTP client for the Healthcare Claims Fraud Detection Environment.

    This client connects to an environment server running via HTTP,
    following the OpenEnv specification.

    Example (connecting to running server):
        >>> client = HealthClaimEnv(base_url="http://localhost:8000")
        >>> obs = client.reset()
        >>> print(f"Prompt: {obs.prompt}")

    Example (starting from Docker):
        >>> client = HealthClaimEnv.from_docker_image("healthcare-claims:latest")
        >>> obs = client.reset()
        >>> client.close()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the environment server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self._container_id: Optional[str] = None

    @classmethod
    def from_docker_image(
        cls,
        image_name: str,
        port: int = 8000,
        host_port: Optional[int] = None,
        timeout: float = 60.0,
    ) -> "HealthClaimEnv":
        """
        Start a Docker container and connect to it.

        Args:
            image_name: Docker image name (e.g., "healthcare-claims:latest")
            port: Container port the server listens on
            host_port: Host port to map to (default: same as container port)
            timeout: Startup timeout in seconds

        Returns:
            Connected HealthClaimEnv client
        """
        host_port = host_port or port

        # Start container
        cmd = ["docker", "run", "-d", "-p", f"{host_port}:{port}", image_name]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        container_id = result.stdout.strip()

        # Create client
        client = cls(base_url=f"http://localhost:{host_port}", timeout=timeout)
        client._container_id = container_id

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = client.session.get(f"{client.base_url}/health", timeout=5)
                if response.status_code == 200:
                    return client
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)

        # Timeout - clean up and raise
        client.close()
        raise RuntimeError(f"Container failed to start within {timeout} seconds")

    def reset(self) -> ClaimObservation:
        """
        Reset the environment and get initial observation.

        Returns:
            ClaimObservation with initial claim data and LLM prompt
        """
        response = self.session.post(f"{self.base_url}/reset", timeout=self.timeout)
        response.raise_for_status()
        return self._parse_observation(response.json())

    def step(self, action: ClaimAction) -> ClaimObservation:
        """
        Execute an action in the environment.

        Args:
            action: The action to take (ClaimAction with response_text)

        Returns:
            ClaimObservation with new claim, reward, done flag
        """
        # Prepare payload - assume Pydantic model
        payload = {"response_text": action.response_text}

        response = self.session.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return self._parse_observation(response.json())

    @property
    def state(self) -> ClaimState:
        """
        Get current episode state.

        Returns:
            ClaimState with episode tracking info
        """
        response = self.session.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        # Handle case where server returns dict
        return ClaimState(**response.json())

    def health_check(self) -> bool:
        """
        Check if the server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_action_space(self) -> Dict[str, Any]:
        """
        Get information about the action space.

        Returns:
            Dictionary describing the action space
        """
        response = self.session.get(f"{self.base_url}/action_space", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get information about the observation space.

        Returns:
            Dictionary describing the observation space
        """
        response = self.session.get(f"{self.base_url}/observation_space", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _parse_observation(self, data: Dict[str, Any]) -> ClaimObservation:
        """Parse JSON response into ClaimObservation."""
        # Handle nested observation if coming from old-style wrapper
        if "observation" in data:
            obs_data = data["observation"]
            obs_data["reward"] = data.get("reward")
            obs_data["done"] = data.get("done")
            obs_data["metadata"] = data.get("info")
            return ClaimObservation(**obs_data)

        # Direct observation
        return ClaimObservation(**data)

    def close(self):
        """
        Clean up resources.

        If started from Docker, stops and removes the container.
        """
        self.session.close()

        if self._container_id:
            # Stop container
            subprocess.run(["docker", "stop", self._container_id], capture_output=True)
            # Remove container
            subprocess.run(["docker", "rm", self._container_id], capture_output=True)
            self._container_id = None

    def __enter__(self) -> "HealthClaimEnv":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.close()
        return False


class LocalHealthClaimEnv:
    """
    Local (non-HTTP) interface to the Claims Fraud Environment.

    Use this for direct Python access without HTTP overhead.
    Useful for development and testing.

    Example:
        >>> from src.envs.healthcare_claims import LocalHealthClaimEnv
        >>> env = LocalHealthClaimEnv()
        >>> obs = env.reset()
        >>> action = ClaimAction(response_text="Decision: APPROVE...")
        >>> obs = env.step(action)
    """

    def __init__(self, **config_kwargs):
        """
        Initialize local environment.

        Args:
            **config_kwargs: Passed to EnvironmentConfig
        """
        from .server.environment import ClaimsFraudEnvironment, EnvironmentConfig

        config = EnvironmentConfig(**config_kwargs) if config_kwargs else None
        self._env = ClaimsFraudEnvironment(config)

    def reset(self) -> ClaimObservation:
        """Reset environment."""
        return self._env.reset()

    def step(self, action: ClaimAction) -> ClaimObservation:
        """Execute action."""
        return self._env.step(action)

    @property
    def state(self) -> ClaimState:
        """Get current state."""
        return self._env.state

    @property
    def action_space_info(self) -> Dict[str, Any]:
        """Get action space info."""
        return self._env.action_space_info

    @property
    def observation_space_info(self) -> Dict[str, Any]:
        """Get observation space info."""
        return self._env.observation_space_info

    def render(self, mode: str = "text") -> Optional[str]:
        """Render environment state."""
        return self._env.render(mode)
