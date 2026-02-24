"""Data update coordinator for Kiwi Voice."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

_LOGGER = logging.getLogger(__name__)

UPDATE_INTERVAL = timedelta(seconds=5)
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=5)
COMMAND_TIMEOUT = aiohttp.ClientTimeout(total=10)


class KiwiVoiceCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator that polls the Kiwi Voice REST API for status updates."""

    def __init__(self, hass: HomeAssistant, host: str, port: int) -> None:
        """Initialize the coordinator.

        Args:
            hass: Home Assistant instance.
            host: Hostname or IP of the Kiwi Voice service.
            port: Port number of the Kiwi Voice REST API.
        """
        super().__init__(
            hass,
            _LOGGER,
            name="Kiwi Voice",
            update_interval=UPDATE_INTERVAL,
        )
        self.base_url = f"http://{host}:{port}/api"
        self._session: aiohttp.ClientSession | None = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Return the shared HTTP session, creating one if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch status, speakers, and languages from the API."""
        session = self._ensure_session()
        try:
            async with session.get(
                f"{self.base_url}/status", timeout=REQUEST_TIMEOUT
            ) as resp:
                resp.raise_for_status()
                status = await resp.json()

            async with session.get(
                f"{self.base_url}/speakers", timeout=REQUEST_TIMEOUT
            ) as resp:
                resp.raise_for_status()
                speakers = await resp.json()

            async with session.get(
                f"{self.base_url}/languages", timeout=REQUEST_TIMEOUT
            ) as resp:
                resp.raise_for_status()
                languages = await resp.json()

            return {
                "status": status,
                "speakers": speakers.get("speakers", []),
                "languages": languages,
            }
        except aiohttp.ClientError as err:
            raise UpdateFailed(
                f"Error communicating with Kiwi Voice API: {err}"
            ) from err
        except Exception as err:
            raise UpdateFailed(
                f"Unexpected error fetching Kiwi Voice data: {err}"
            ) from err

    async def async_send_command(
        self,
        endpoint: str,
        method: str = "POST",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a command to the Kiwi Voice API.

        Args:
            endpoint: API endpoint path (e.g. "stop", "tts/test").
            method: HTTP method (POST, PATCH, etc.).
            data: Optional JSON body.

        Returns:
            Parsed JSON response from the API.
        """
        session = self._ensure_session()
        url = f"{self.base_url}/{endpoint}"
        async with session.request(
            method, url, json=data, timeout=COMMAND_TIMEOUT
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def async_shutdown(self) -> None:
        """Close the HTTP session on coordinator shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
