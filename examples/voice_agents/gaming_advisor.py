import logging
from typing import List, Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("gaming-advisor")

load_dotenv()


class GamingAdvisorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are NexusGuide, an advanced AI gaming assistant from the future.

Your purpose is to provide expert gaming advice, recommendations, and assistance to players.
Your tone is friendly, enthusiastic, and knowledgeable - like the ultimate gaming buddy.

Some key personality traits:
- You have extensive knowledge of video games from all eras (classic to futuristic)
- You're passionate about gaming culture and esports
- You provide strategic advice without being condescending
- You can recommend games based on player preferences
- You can troubleshoot common gaming issues
- You have a good sense of humor and occasionally make gaming-related jokes
- You keep responses concise and conversational since this is a voice interface

You can assist with game recommendations, strategies, Easter eggs, achievement hunting,
hardware advice, and more. When you don't know something specific, you'll be honest
but try to provide general guidance that might help.
""",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply(
            instructions="Greet the user enthusiastically as NexusGuide, the future of gaming advice, and ask how you can help them with their gaming needs today."
        )

    @function_tool
    async def recommend_games(
        self,
        context: RunContext,
        genre: Optional[str] = None,
        platform: Optional[str] = None,
        multiplayer: Optional[bool] = None,
        similar_to: Optional[str] = None,
    ):
        """Recommends video games based on user preferences.

        Args:
            genre: The genre of games the user is interested in (e.g., "FPS", "RPG", "Strategy")
            platform: The gaming platform (e.g., "PC", "PlayStation", "Xbox", "Switch", "Mobile")
            multiplayer: Whether the user wants multiplayer games
            similar_to: A game that the user already enjoys, to find similar recommendations
        """

        logger.info(f"Recommending games with parameters: genre={genre}, platform={platform}, multiplayer={multiplayer}, similar_to={similar_to}")

        # In a real implementation, this would query a database or API
        # For this example, we'll return a structured response that the LLM can use
        return {
            "recommendations": [
                {
                    "title": "Stellar Odyssey",
                    "genre": "Space RPG",
                    "description": "An immersive open-world space exploration game with deep character progression",
                    "platforms": ["PC", "PlayStation", "Xbox"],
                    "multiplayer": True,
                    "rating": 9.2,
                },
                {
                    "title": "Neon Breach",
                    "genre": "Cyberpunk FPS",
                    "description": "Fast-paced shooter set in a dystopian future with unique hacking mechanics",
                    "platforms": ["PC", "PlayStation", "Xbox"],
                    "multiplayer": True,
                    "rating": 8.8,
                },
                {
                    "title": "Echo Realm",
                    "genre": "Puzzle Adventure",
                    "description": "Mind-bending puzzle game where sound and music control the environment",
                    "platforms": ["PC", "Switch", "Mobile"],
                    "multiplayer": False,
                    "rating": 9.0,
                },
            ],
            "notes": "These recommendations are based on your preferences. I can provide more specific suggestions if you tell me more about what you enjoy in games."
        }

    @function_tool
    async def provide_strategy(
        self,
        context: RunContext,
        game: str,
        specific_challenge: Optional[str] = None,
        character_class: Optional[str] = None,
        difficulty: Optional[str] = None,
    ):
        """Provides gaming strategies and tips for specific games or challenges.

        Args:
            game: The name of the game the user needs help with
            specific_challenge: A specific level, boss, achievement or challenge they're stuck on
            character_class: If applicable, the character class or build they're using
            difficulty: The difficulty level they're playing on
        """

        logger.info(f"Providing strategy for: game={game}, challenge={specific_challenge}, class={character_class}, difficulty={difficulty}")

        # In a real implementation, this would query a database or API
        return {
            "game": game,
            "strategy": f"Here's a strategic approach for {game}" + 
                      (f" when facing {specific_challenge}" if specific_challenge else "") +
                      (f" using {character_class}" if character_class else "") +
                      (f" on {difficulty} difficulty" if difficulty else "") +
                      ":\n\n" +
                      "1. Start by analyzing the pattern of the challenge\n" +
                      "2. Ensure your equipment is optimized for this specific encounter\n" +
                      "3. Consider adjusting your timing rather than being aggressive\n" +
                      "4. Look for environmental advantages you might have missed",
            "additional_tips": [
                "Remember that patience is often key to overcoming difficult challenges",
                "The community has found that upgrading your defensive capabilities helps significantly",
                "There might be optional quests that provide items specifically designed for this challenge"
            ]
        }

    @function_tool
    async def troubleshoot_technical_issue(
        self,
        context: RunContext,
        hardware: str,
        game: Optional[str] = None,
        symptoms: str = "",
        tried_solutions: Optional[List[str]] = None,
    ):
        """Helps troubleshoot technical gaming issues.

        Args:
            hardware: The gaming hardware experiencing issues (console name, PC specs, etc.)
            game: The specific game having problems, if applicable
            symptoms: Description of the technical issues being experienced
            tried_solutions: Solutions the user has already attempted
        """

        logger.info(f"Troubleshooting: hardware={hardware}, game={game}, symptoms={symptoms}")

        return {
            "possible_causes": [
                "Outdated drivers or system software",
                "Insufficient system resources for game requirements",
                "Corrupted game files or installation",
                "Hardware compatibility issues",
                "Network connectivity problems (for online features)"
            ],
            "recommended_solutions": [
                "Update all drivers and system software to the latest version",
                "Verify and repair game files through the launcher/store",
                "Check for background applications consuming resources",
                "Adjust in-game graphics settings to better match your hardware",
                "Try a clean reinstallation if other solutions don't work"
            ],
            "preventative_tips": [
                "Regularly update drivers and system software",
                "Monitor system temperatures during gaming sessions",
                "Keep storage drives uncrowded with at least 15-20% free space",
                "Consider hardware upgrades if you're frequently encountering performance issues"
            ]
        }


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "gaming_advisor_user",
    }
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # Using OpenAI's latest models for best gaming knowledge
        llm=openai.LLM(model="gpt-4o"),  # Using the more capable model for gaming knowledge
        stt=openai.STT(model="whisper-1"),  # Good for recognizing gaming terminology
        tts=openai.TTS(voice="nova", model="tts-1"),  # Energetic voice that fits gaming context
        turn_detection=MultilingualModel(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    # wait for a participant to join the room
    await ctx.wait_for_participant()

    await session.start(
        agent=GamingAdvisorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))