"""
navigation/gps_navigator.py — GPS-Based Navigation Engine
===========================================================
Provides turn-by-turn navigation from current GPS position to a destination.

Two modes:
  1. Online — Nominatim geocoding + OpenStreetMap routing (osmnx)
  2. Offline — Cached OSM graph + straight-line bearing navigation

Navigation instructions:
  • "Turn left in 30 metres"
  • "Walk straight for 100 metres"
  • "You have arrived at your destination"
  • "Recalculating route…"

Provides:
  • async navigate_to(destination: str) → bool
  • async stop_navigation()
  • async on_gps_fix(fix)               — called by main loop
  • async initialise() / cleanup()
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from sensors.gps_sensor import GPSFix
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Waypoint:
    latitude: float
    longitude: float
    instruction: str = ""
    distance_m: float = 0.0

    def to_fix(self) -> GPSFix:
        return GPSFix(latitude=self.latitude, longitude=self.longitude, is_valid=True)


class GPSNavigator:
    """
    GPS turn-by-turn navigator with OSM geocoding and offline fallback.
    """

    def __init__(self, cfg: Config, gps_module):
        self.cfg = cfg
        self._gps = gps_module
        self._current_fix: Optional[GPSFix] = None
        self._route: list[Waypoint] = []
        self._current_waypoint_idx = 0
        self._navigating = False
        self._destination_name = ""
        self._tts = None   # injected after init (circular dep avoided)
        self._last_instruction_time = 0.0
        self._osm_graph = None

    def set_tts(self, tts):
        """Inject TTS after construction to avoid circular dependency."""
        self._tts = tts

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        if not self.cfg.system.ENABLE_GPS:
            log.info("Navigator: GPS disabled.")
            return
        log.info("GPS navigator ready.")

    async def cleanup(self):
        self._navigating = False
        self._route.clear()

    # ── Navigation control ────────────────────────────────────────────────────
    async def navigate_to(self, destination: str) -> bool:
        """
        Geocode destination and build a route.
        Returns True if route was found.
        """
        self._destination_name = destination

        # Geocode the destination
        dest_fix = await self._geocode(destination)
        if not dest_fix:
            log.warning(f"Could not geocode: {destination}")
            return False

        # Build route waypoints
        route = await self._build_route(self._current_fix, dest_fix, destination)
        if not route:
            # Fallback: direct bearing (no waypoints)
            route = self._direct_route(self._current_fix, dest_fix, destination)

        self._route = route
        self._current_waypoint_idx = 0
        self._navigating = True

        log.info(f"Route to '{destination}': {len(route)} waypoints.")

        # Announce first instruction
        if route and self._tts:
            await self._tts.speak(route[0].instruction, priority="NAVIGATION")

        return True

    async def stop_navigation(self):
        self._navigating = False
        self._route.clear()
        self._current_waypoint_idx = 0
        log.info("Navigation stopped.")

    # ── GPS fix handler ───────────────────────────────────────────────────────
    async def on_gps_fix(self, fix: GPSFix):
        """Called with each new GPS fix. Updates position and checks route progress."""
        self._current_fix = fix

        if not self._navigating or not self._route:
            return

        now = time.time()
        min_interval = self.cfg.navigation.NAV_RECALCULATE_INTERVAL

        # Check if we've reached the current waypoint
        if self._current_waypoint_idx < len(self._route):
            wp = self._route[self._current_waypoint_idx]
            dist = fix.distance_to(wp.to_fix())

            if dist < 15.0:   # within 15m of waypoint → advance
                self._current_waypoint_idx += 1
                if self._current_waypoint_idx >= len(self._route):
                    await self._announce_arrival()
                    return
                next_wp = self._route[self._current_waypoint_idx]
                if self._tts and now - self._last_instruction_time > 5.0:
                    self._last_instruction_time = now
                    await self._tts.speak(next_wp.instruction, priority="NAVIGATION")

            elif now - self._last_instruction_time > min_interval:
                # Periodic distance update
                self._last_instruction_time = now
                bearing = fix.bearing_to(wp.to_fix())
                direction = self._bearing_to_direction(bearing, fix.bearing_deg)
                if self._tts:
                    await self._tts.speak(
                        f"{int(dist)} metres to {self._destination_name}. {direction}.",
                        priority="NAVIGATION",
                    )

    async def _announce_arrival(self):
        self._navigating = False
        if self._tts:
            await self._tts.speak(
                f"You have arrived at {self._destination_name}.",
                priority="NAVIGATION",
            )
        log.info(f"Arrived at destination: {self._destination_name}")

    # ── Geocoding ─────────────────────────────────────────────────────────────
    async def _geocode(self, place: str) -> Optional[GPSFix]:
        """Geocode place name to GPS coordinates using Nominatim."""
        loop = asyncio.get_event_loop()
        try:
            from geopy.geocoders import Nominatim
            geocoder = Nominatim(user_agent=self.cfg.navigation.NOMINATIM_USER_AGENT)
            location = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: geocoder.geocode(place)),
                timeout=10.0,
            )
            if location:
                log.info(f"Geocoded '{place}' → ({location.latitude}, {location.longitude})")
                return GPSFix(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    is_valid=True,
                )
        except Exception as e:
            log.warning(f"Geocoding error: {e}")
        return None

    # ── Route building ────────────────────────────────────────────────────────
    async def _build_route(
        self,
        origin: Optional[GPSFix],
        dest: GPSFix,
        destination_name: str,
    ) -> list[Waypoint]:
        """Build walking route using osmnx + networkx."""
        if origin is None or not origin.has_fix:
            return []
        loop = asyncio.get_event_loop()
        try:
            route = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._osmnx_route(origin, dest, destination_name)
                ),
                timeout=30.0,
            )
            return route
        except asyncio.TimeoutError:
            log.warning("OSMnx route timed out — using direct bearing.")
            return []
        except Exception as e:
            log.warning(f"OSMnx route error: {e}")
            return []

    def _osmnx_route(
        self,
        origin: GPSFix,
        dest: GPSFix,
        destination_name: str,
    ) -> list[Waypoint]:
        """Blocking OSMnx-based walking route calculation."""
        import osmnx as ox
        import networkx as nx

        cache_dir = str(self.cfg.navigation.OSM_CACHE_DIR)
        ox.settings.use_cache = True
        ox.settings.cache_folder = cache_dir

        # Download graph around midpoint
        mid_lat = (origin.latitude + dest.latitude) / 2
        mid_lon = (origin.longitude + dest.longitude) / 2
        dist = origin.distance_to(dest)
        radius = max(500, dist * 0.7)

        G = ox.graph_from_point((mid_lat, mid_lon), dist=radius, network_type="walk")

        orig_node = ox.nearest_nodes(G, origin.longitude, origin.latitude)
        dest_node = ox.nearest_nodes(G, dest.longitude, dest.latitude)

        path_nodes = nx.shortest_path(G, orig_node, dest_node, weight="length")

        waypoints = []
        for i, node_id in enumerate(path_nodes):
            node = G.nodes[node_id]
            lat, lon = node["y"], node["x"]

            if i == 0:
                instruction = f"Starting navigation to {destination_name}."
            elif i == len(path_nodes) - 1:
                instruction = f"You have arrived at {destination_name}."
            else:
                # Calculate turn instruction
                prev_node = G.nodes[path_nodes[i - 1]]
                bearing = self._calc_bearing(
                    prev_node["y"], prev_node["x"], lat, lon
                )
                instruction = f"Continue {self._bearing_label(bearing)} for {int(dist)} metres."

            waypoints.append(Waypoint(
                latitude=lat,
                longitude=lon,
                instruction=instruction,
            ))

        return waypoints

    def _direct_route(
        self,
        origin: Optional[GPSFix],
        dest: GPSFix,
        destination_name: str,
    ) -> list[Waypoint]:
        """Fallback: two-waypoint direct bearing route."""
        if origin is None:
            return [Waypoint(
                latitude=dest.latitude,
                longitude=dest.longitude,
                instruction=f"Head toward {destination_name}.",
            )]
        bearing = origin.bearing_to(dest)
        dist = origin.distance_to(dest)
        return [
            Waypoint(
                latitude=dest.latitude,
                longitude=dest.longitude,
                instruction=(
                    f"Walk {self._bearing_label(bearing)} for "
                    f"approximately {int(dist)} metres to reach {destination_name}."
                ),
                distance_m=dist,
            )
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_bearing(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    @staticmethod
    def _bearing_label(bearing: float) -> str:
        dirs = ["north", "north-east", "east", "south-east",
                "south", "south-west", "west", "north-west"]
        idx = round(bearing / 45) % 8
        return dirs[idx]

    @staticmethod
    def _bearing_to_direction(target_bearing: float, current_bearing: float) -> str:
        """Convert target bearing relative to user heading into a turn instruction."""
        diff = (target_bearing - current_bearing + 360) % 360
        if diff < 30 or diff > 330:
            return "Continue straight ahead"
        elif diff < 90:
            return "Turn slightly right"
        elif diff < 180:
            return "Turn right"
        elif diff < 270:
            return "Turn left"
        else:
            return "Turn slightly left"

    def is_ready(self) -> bool:
        return True

    @property
    def is_navigating(self) -> bool:
        return self._navigating
