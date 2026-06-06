"""
ai/decision_engine.py — Fused AI Decision Engine & Companion Brain
===================================================================
Fuses inputs from all sensors (camera, ultrasonic, GPS) and AI models 
to drive a real-time conversational companion for visually impaired users.

Key Features:
  1. Sensory Fusion Compiler: Continuously maps surroundings, faces, text, GPS.
  2. Proactive Guidance Loop: Background loop providing natural navigation and safety hints.
  3. Conversational Ollama Engine: Route voice handlers to Ollama with full context.
  4. Robust Rule-based Fallback: Safe offline interpretation when Ollama is offline.
"""

import asyncio
import base64
import time
from typing import Optional, List

import cv2
import numpy as np

from config import Config
from utils.logger import get_logger
from ai.intent_parser import IntentParser, IntentType
from sensors.ultrasonic import CRITICAL_CM, WARNING_CM

log = get_logger(__name__)


class DecisionEngine:
    """Master decision engine — fuses all sensor/AI inputs and drives TTS output."""

    def __init__(
        self, cfg, tts, llm, intent, detector,
        scene_analyzer, face_recognizer, currency_detector,
        navigator, memory, sos
    ):
        self.cfg            = cfg
        self._tts           = tts
        self._llm           = llm
        self._intent        = intent
        self._detector      = detector
        self._scene         = scene_analyzer
        self._face_rec      = face_recognizer
        self._currency      = currency_detector
        self._navigator     = navigator
        self._memory        = memory
        self._sos           = sos

        # Dynamic State Memory
        self._last_distance = 999.0
        self._last_dist_alert_time  = 0.0
        self._last_guidance_time    = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self._current_gps_fix = None
        self._announced_faces: dict[str, float] = {}   # name → last announced
        self._last_ocr_text: str = ""
        
        # State tracking for proactive updates
        self._last_announced_state = {
            "obstacle_priority": "CLEAR",
            "hazards_count": 0,
            "next_waypoint_idx": -1,
            "seen_faces": set(),
        }
        
        self._proactive_task: Optional[asyncio.Task] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        self._running = True
        self._proactive_task = asyncio.create_task(self._proactive_guidance_loop())
        log.info("Decision engine ready. Proactive guidance loop started.")

    async def cleanup(self):
        self._running = False
        if self._proactive_task:
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except asyncio.CancelledError:
                pass
        log.info("Decision engine stopped.")

    def is_ready(self) -> bool:
        return True

    # ── Sensor Input Handlers ──────────────────────────────────────────────────
    async def on_distance(self, distance_cm: float):
        """Called every time the ultrasonic sensor produces a reading."""
        now = time.time()
        self._last_distance = distance_cm

        # EMERGENCY OVERRIDE: instant warning bypassing Ollama to protect the user
        if distance_cm < CRITICAL_CM:
            if now - self._last_dist_alert_time > 3.0:
                self._last_dist_alert_time = now
                await self._tts.speak(
                    f"Stop! Obstacle just {int(distance_cm)} centimetres ahead!",
                    priority="EMERGENCY",
                )

    async def on_camera_frame(self, frame: np.ndarray):
        """Called for every camera frame."""
        self._last_frame = frame
        log.debug(f"[Pipeline] Camera frame captured. Shape: {frame.shape}")
        # Keep face recognizer registry running
        await self._check_faces(frame)

    async def on_gps_fix(self, fix):
        """Update current GPS position."""
        self._current_gps_fix = fix
        log.debug(f"[Pipeline] GPS fix updated: {fix}")

    async def _check_faces(self, frame: np.ndarray):
        """Analyze faces in frame for proactive updates."""
        if not self.cfg.system.ENABLE_FACE_RECOGNITION:
            return
        # Process frame internally (face recognizer throttles itself to 2 FPS)
        await self._face_rec.process_frame(frame)

    # ── Sensory Fusion Compiler ───────────────────────────────────────────────
    def _get_fused_state(self) -> dict:
        """Compile a complete structured dictionary of the user's current environment state."""
        # Obstacle assessment
        dist = self._last_distance
        dist_priority = "CLEAR"
        if dist < CRITICAL_CM:
            dist_priority = "CRITICAL"
        elif dist < WARNING_CM:
            dist_priority = "WARNING"

        # YOLO object detections
        loop = asyncio.get_event_loop()
        detections = []
        if self._detector.is_ready():
            # Run get_latest_detections (non-blocking lookup)
            try:
                raw_dets = asyncio.run_coroutine_threadsafe(
                    self._detector.get_latest_detections(), loop
                ).result(timeout=0.01)
                for d in raw_dets:
                    detections.append({
                        "class_name": d.class_name,
                        "position": d.position_relative,
                        "distance_m": d.estimated_distance_m,
                    })
            except Exception:
                pass

        # GPS & Navigation
        gps_state = {"has_fix": False, "lat": 0.0, "lon": 0.0}
        if self._current_gps_fix and self._current_gps_fix.has_fix:
            gps_state = {
                "has_fix": True,
                "lat": self._current_gps_fix.latitude,
                "lon": self._current_gps_fix.longitude,
                "speed_kmh": self._current_gps_fix.speed_kmh,
            }

        nav_state = {"is_navigating": False}
        if self._navigator.is_navigating:
            nav_state = {
                "is_navigating": True,
                "destination": self._navigator._destination_name,
                "current_waypoint_idx": self._navigator._current_waypoint_idx,
                "total_waypoints": len(self._navigator._route),
            }
            if self._navigator._current_waypoint_idx < len(self._navigator._route):
                wp = self._navigator._route[self._navigator._current_waypoint_idx]
                nav_state["next_instruction"] = wp.instruction
                if self._current_gps_fix:
                    nav_state["distance_to_waypoint_m"] = int(self._current_gps_fix.distance_to(wp.to_fix()))

        # Faces detected
        faces = []
        for face in self._face_rec._last_results:
            faces.append({"name": face.name, "known": face.is_known})

        fused = {
            "ultrasonic": {
                "distance_cm": dist,
                "priority": dist_priority,
            },
            "detections": detections,
            "gps": gps_state,
            "navigation": nav_state,
            "faces": faces,
            "ocr_text": self._last_ocr_text,
        }
        log.debug(f"[Pipeline] Compiled fused sensory state: {fused}")
        return fused

    # ── Proactive Guidance Loop ───────────────────────────────────────────────
    async def _proactive_guidance_loop(self):
        """Continuously monitors for environmental changes and generates natural spoken guidance."""
        # Wait for systems to warm up
        await asyncio.sleep(4.0)

        while self._running:
            try:
                state = self._get_fused_state()
                now = time.time()
                
                # Check for triggers:
                # 1. Obstacle priority changed (e.g. CLEAR -> WARNING)
                # 2. Vehicle or person entered walking path (center)
                # 3. GPS navigation step advanced
                # 4. New face detected
                # 5. Ambient timeout (IDLE_DESCRIPTION_INTERVAL) reached
                
                trigger = False
                trigger_reason = ""

                # Trigger 1: Distance priority change
                current_prio = state["ultrasonic"]["priority"]
                if current_prio != self._last_announced_state["obstacle_priority"]:
                    if current_prio in ("WARNING", "CRITICAL"):
                        trigger = True
                        trigger_reason = f"Distance warning level went to {current_prio}"
                    self._last_announced_state["obstacle_priority"] = current_prio

                # Trigger 2: New hazards in path
                hazards = [d for d in state["detections"] if d["class_name"] in ("person", "car", "bus", "truck", "motorcycle", "stairs") and d["position"] == "center"]
                if len(hazards) != self._last_announced_state["hazards_count"]:
                    if len(hazards) > self._last_announced_state["hazards_count"]:
                        trigger = True
                        trigger_reason = f"New hazard detected in path: {hazards[0]['class_name']}"
                    self._last_announced_state["hazards_count"] = len(hazards)

                # Trigger 3: Navigation waypoint index changes
                if state["navigation"]["is_navigating"]:
                    current_idx = state["navigation"]["current_waypoint_idx"]
                    if current_idx != self._last_announced_state["next_waypoint_idx"]:
                        trigger = True
                        trigger_reason = "GPS waypoint advanced"
                        self._last_announced_state["next_waypoint_idx"] = current_idx

                # Trigger 4: Face announcements
                current_faces = {f["name"] for f in state["faces"] if f["known"]}
                new_faces = current_faces - self._last_announced_state["seen_faces"]
                if new_faces:
                    trigger = True
                    trigger_reason = f"New face detected: {list(new_faces)}"
                    self._last_announced_state["seen_faces"] = current_faces

                # Trigger 5: Ambient time-out
                if now - self._last_guidance_time > self.cfg.system.IDLE_DESCRIPTION_INTERVAL:
                    trigger = True
                    trigger_reason = "Idle ambient interval elapsed"

                # If triggered, enforce cooldown
                cooldown = self.cfg.system.PROACTIVE_SAFETY_COOLDOWN
                # Tighten cooldown if there is a critical change
                if trigger_reason.startswith("New hazard") or "warning" in trigger_reason:
                    cooldown = 3.0

                if trigger and (now - self._last_guidance_time > cooldown):
                    log.info(f"[Pipeline] Proactive guidance trigger fired! Reason: '{trigger_reason}'")
                    self._last_guidance_time = now
                    await self._generate_guidance(state, trigger_reason)

            except Exception as e:
                log.error(f"Error in proactive guidance loop: {e}")

            await asyncio.sleep(1.0)

    async def _generate_guidance(self, state: dict, reason: str):
        """Call Ollama (or fallback) to formulate natural, human-like guide instructions."""
        log.debug(f"[Pipeline] Formulating proactive guidance. Reason: '{reason}'")

        if self._llm.is_ready():
            prompt = (
                "You are an AI companion guiding a blind person. Based on this sensor state, "
                "generate a extremely brief (max 1 sentence) spoken guidance statement. "
                "Speak directly to the user (e.g. 'A person is walking towards you from the center path' "
                "or 'Move slightly left, the path ahead is clear' or 'Stop, a car is on your right'). "
                "Only output the spoken response. If the environment is clear and normal, output '[NO_UPDATE]'.\n\n"
                f"State: {state}"
            )
            log.info(f"[Pipeline] Querying Ollama (model={self.cfg.ai.OLLAMA_TEXT_MODEL}) with proactive prompt: '{prompt}'")
            try:
                response = await self._llm.query(prompt)
                response = response.strip()
                log.info(f"[Pipeline] Ollama response: '{response}'")
                if response and response != "[NO_UPDATE]":
                    log.info(f"[Pipeline] Queuing TTS speech: '{response}'")
                    await self._tts.speak(response, priority="NAVIGATION")
                    return
                else:
                    log.debug("[Pipeline] Ollama requested NO_UPDATE. Falling through to fallback generator.")
            except Exception as e:
                log.warning(f"Ollama proactive guidance call failed: {e}")

        # Graceful rule-based fallback guide description
        fallback_msg = self._generate_rule_based_fallback(state)
        if fallback_msg:
            log.info(f"[Pipeline] rule fallback summary generated: '{fallback_msg}'")
            await self._tts.speak(fallback_msg, priority="NAVIGATION")

    def _generate_rule_based_fallback(self, state: dict) -> Optional[str]:
        """Formulate direct guidance instructions from state without an LLM."""
        # 1. Obstacle warnings
        if state["ultrasonic"]["priority"] == "CRITICAL":
            return f"Stop. Obstacle detected very close, just {int(state['ultrasonic']['distance_cm'])} centimetres away."
        elif state["ultrasonic"]["priority"] == "WARNING":
            return f"Caution, obstacle {int(state['ultrasonic']['distance_cm'])} centimetres ahead."
        
        # 2. Hazards in path
        path_hazards = [d for d in state["detections"] if d["class_name"] in ("car", "truck", "bus", "motorcycle", "person", "stairs")]
        if path_hazards:
            center_hazards = [d for d in path_hazards if d["position"] == "center"]
            if center_hazards:
                h = center_hazards[0]
                return f"Caution. A {h['class_name']} is directly in front of you, approximately {h['distance_m']} meters ahead."
            else:
                h = path_hazards[0]
                return f"Note. A {h['class_name']} is on your {h['position']}, about {h['distance_m']} meters away."

        # 3. Face recognition
        known_faces = [f["name"] for f in state["faces"] if f["known"]]
        if known_faces:
            return f"{', '.join(known_faces)} is in front of you."

        # 4. Active GPS instructions
        if state["navigation"]["is_navigating"] and "next_instruction" in state["navigation"]:
            nav = state["navigation"]
            return f"Navigation: {nav['next_instruction']}. {nav.get('distance_to_waypoint_m', 0)} meters remaining to turn."

        # 5. Default ambient summary (ensures continuous guidance announcements)
        dets = state["detections"]
        if dets:
            desc = ", ".join(f"a {d['class_name']} on your {d['position']}" for d in dets[:2])
            return f"I observe {desc} ahead. The path forward is open."
        
        return "The path ahead is clear."

    # ── Voice Input Intent Handlers ───────────────────────────────────────────
    async def on_voice_input(self, text: str):
        """Route voice commands to appropriate handler based on intent."""
        if not text or len(text.strip()) < 2:
            return

        log.info(f"Processing voice command: '{text}'")
        intent = await self._intent.classify(text)
        log.info(f"Classified Intent: {intent.type.value} (confidence={intent.confidence:.0%})")

        handlers = {
            IntentType.DESCRIBE:   self._handle_describe,
            IntentType.NAVIGATE:   self._handle_navigate,
            IntentType.READ:       self._handle_read,
            IntentType.FACE:       self._handle_face,
            IntentType.CURRENCY:   self._handle_currency,
            IntentType.EMERGENCY:  self._handle_emergency,
            IntentType.REMEMBER:   self._handle_remember,
            IntentType.STOP_NAV:   self._handle_stop_nav,
            IntentType.STATUS:     self._handle_status,
            IntentType.UNKNOWN:    self._handle_unknown,
        }

        handler = handlers.get(intent.type, self._handle_unknown)
        await handler(intent)

    async def _handle_describe(self, intent):
        """Describe the current scene using LLM/Florence-2 incorporating sensory data."""
        await self._tts.speak("Analysing your surroundings.", priority="INFO")

        frame = self._last_frame
        if frame is None:
            await self._tts.speak("Camera feed not available. I cannot describe the scene.", priority="INFO")
            return

        state = self._get_fused_state()
        detections = await self._detector.get_latest_detections()
        analysis = await self._scene.analyse(frame, query=intent.text, detections=detections)

        # Feed the description result through Ollama for natural conversational phrasing
        if self._llm.is_ready():
            prompt = (
                f"The user is asking: '{intent.text}'. Here is the sensor state: {state}. "
                f"And here is the visual analysis of the camera: '{analysis.caption}'. "
                "Synthesize this into a natural, conversational description as a human guide. "
                "Limit your answer to 2 or 3 sentences. Be actionable and clear."
            )
            response = await self._llm.query(prompt)
        else:
            response = analysis.caption or analysis.navigation_hint
            if not response:
                response = "I see a clear path ahead with no immediate obstacles."

        await self._tts.speak(response, priority="INFO")
        await self._memory.store_scene(response, self._current_gps_fix)

    async def _handle_navigate(self, intent):
        """Start GPS navigation to a destination."""
        destination = intent.entity
        if not destination:
            await self._tts.speak(
                "Please state where you'd like to navigate. For example, 'take me to the bus stop'.",
                priority="INFO",
            )
            return

        await self._tts.speak(f"Finding route to {destination}.", priority="NAVIGATION")
        success = await self._navigator.navigate_to(destination)
        if not success:
            await self._tts.speak(f"Sorry, I could not resolve a route to {destination}.", priority="INFO")
        else:
            if self._llm.is_ready():
                prompt = (
                    f"The user has requested navigation to '{destination}'. A route has been found. "
                    "Confirm this to the user naturally in 1 sentence, encouraging them to walk forward."
                )
                response = await self._llm.query(prompt)
                await self._tts.speak(response, priority="NAVIGATION")

    async def _handle_read(self, intent):
        """Read text in front of the camera using OCR."""
        from ocr.text_reader import TextReader
        frame = self._last_frame
        if frame is None:
            await self._tts.speak("Camera not available.", priority="INFO")
            return

        await self._tts.speak("Reading text. Hold steady.", priority="INFO")
        reader = TextReader(self.cfg)
        await reader.initialise()
        text = await reader.read_frame(frame)

        if text:
            self._last_ocr_text = text
            if self._llm.is_ready():
                prompt = (
                    f"The camera OCR read this text: '{text}'. "
                    "Read this text clearly to the user, summarizing or highlighting key details "
                    "if it looks like a sign, label, or storefront. Keep it short."
                )
                response = await self._llm.query(prompt)
                await self._tts.speak(response, priority="INFO")
            else:
                await self._tts.speak(f"It says: {text}", priority="INFO")
        else:
            await self._tts.speak("No readable text was found in view.", priority="INFO")

    async def _handle_face(self, intent):
        """Identify person(s) in frame."""
        frame = self._last_frame
        if frame is None:
            await self._tts.speak("Camera feed offline.", priority="INFO")
            return

        faces = await self._face_rec.process_frame(frame)
        if not faces:
            await self._tts.speak("No person detected in front of you.", priority="INFO")
            return

        known = [f.name for f in faces if f.is_known]
        unknown_count = len(faces) - len(known)

        if self._llm.is_ready():
            prompt = (
                f"I detected these faces: known names: {known}, unknown count: {unknown_count}. "
                "Converse naturally with the user, telling them who is there. Keep it friendly."
            )
            response = await self._llm.query(prompt)
            await self._tts.speak(response, priority="INFO")
        else:
            parts = []
            if known:
                parts.append(", ".join(known))
            if unknown_count:
                parts.append(f"{unknown_count} unknown person")
            await self._tts.speak(f"I can see {' and '.join(parts)} in front of you.", priority="INFO")

    async def _handle_currency(self, intent):
        """Detect and announce currency denomination."""
        frame = self._last_frame
        if frame is None:
            await self._tts.speak("Camera not available.", priority="INFO")
            return

        await self._tts.speak("Checking note. Hold it flat.", priority="INFO")
        result = await self._currency.detect(frame)

        if result:
            if self._llm.is_ready():
                prompt = (
                    f"The detector identified a currency note: {result.denomination} {result.currency}. "
                    "Announce this to the user naturally in a brief sentence."
                )
                response = await self._llm.query(prompt)
                await self._tts.speak(response, priority="INFO")
            else:
                await self._tts.speak(f"This is a {result.denomination} {result.currency} note.", priority="INFO")
        else:
            await self._tts.speak("I could not identify any currency note in view.", priority="INFO")

    async def _handle_emergency(self, intent):
        """Trigger SOS protocol."""
        await self._sos.trigger(
            message=f"User voice trigger: '{intent.text}'",
            gps_fix=self._current_gps_fix,
        )

    async def _handle_remember(self, intent):
        """Enroll a new face with a given name."""
        name = intent.entity
        if not name:
            await self._tts.speak("Please repeat with a name, for example: 'Remember this person as John'.", priority="INFO")
            return

        frame = self._last_frame
        if frame is None:
            await self._tts.speak("Camera not available.", priority="INFO")
            return

        await self._tts.speak(f"Registering face for {name}.", priority="INFO")
        success = await self._face_rec.enroll_face(name, frame)
        if success:
            await self._tts.speak(f"I have saved the face. I will remember this person as {name}.", priority="INFO")
        else:
            await self._tts.speak("I could not detect a face clearly. Try centering the person in the camera.", priority="INFO")

    async def _handle_stop_nav(self, intent):
        """Stop active navigation."""
        await self._navigator.stop_navigation()
        await self._tts.speak("Navigation stopped.", priority="NAVIGATION")

    async def _handle_status(self, intent):
        """Report system status."""
        state = self._get_fused_state()
        if self._llm.is_ready():
            prompt = (
                f"Translate this system state dict into a friendly, spoken status update: {state}. "
                "Mention battery/reasoning status, GPS position, and nearest obstacle distance. Keep it snazzy."
            )
            response = await self._llm.query(prompt)
            await self._tts.speak(response, priority="INFO")
        else:
            parts = [
                "AI reasoning is offline" if not self._llm.is_ready() else "AI reasoning online",
                "GPS fixed" if state["gps"]["has_fix"] else "GPS searching",
                f"nearest obstacle {int(self._last_distance)} centimetres",
            ]
            await self._tts.speak("System status: " + ", ".join(parts) + ".", priority="INFO")

    async def _handle_unknown(self, intent):
        """Handle unknown queries by querying Ollama with full sensory context."""
        state = self._get_fused_state()

        if self._llm.is_ready():
            prompt = (
                f"You are a real-time conversational AI guide for a visually impaired user. "
                f"The user is asking: '{intent.text}'. "
                f"Here is the complete sensor state of their current environment: {state}. "
                "Answer their question directly and conversationally as a human companion would. "
                "If they ask if they can cross the road, look for vehicles or green traffic lights in the detections. "
                "If they ask where they are, refer to GPS coordinates. Be helpful, concise, and prioritize safety."
            )
            response = await self._llm.chat(prompt)
            await self._tts.speak(response, priority="INFO")
        else:
            # Rule based keyword answers if offline
            text_lower = intent.text.lower()
            if "cross" in text_lower or "road" in text_lower:
                # Check detections for cars
                vehicles = [d for d in state["detections"] if d["class_name"] in ("car", "bus", "truck", "motorcycle")]
                if vehicles:
                    await self._tts.speak("I detect vehicles nearby. It may not be safe to cross. Please proceed with caution.", priority="HIGH")
                else:
                    await self._tts.speak("I don't see any vehicles on the path. You may proceed, but listen carefully.", priority="HIGH")
            elif "where" in text_lower or "location" in text_lower:
                if state["gps"]["has_fix"]:
                    await self._tts.speak(f"You are at coordinates latitude {state['gps']['lat']:.4f}, longitude {state['gps']['lon']:.4f}.", priority="INFO")
                else:
                    await self._tts.speak("GPS signal is weak, searching for position.", priority="INFO")
            else:
                await self._tts.speak("I didn't catch that. Please repeat your question.", priority="INFO")
