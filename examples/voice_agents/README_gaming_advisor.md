# NexusGuide: Futuristic Gaming Advisor Agent

NexusGuide is an advanced voice-based AI gaming assistant built with LiveKit Agents. It provides expert gaming advice, recommendations, and assistance to players - like a futuristic version of the classic Nintendo game helpline, but with modern capabilities.

## Features

- **Game Recommendations**: Suggests games based on genre, platform preferences, multiplayer options, or similarity to games the user already enjoys
- **Strategic Gaming Advice**: Provides tips, walkthroughs, and strategies for specific games, challenges, character classes, and difficulty levels
- **Technical Troubleshooting**: Helps diagnose and solve hardware and software gaming issues
- **Future-Forward Persona**: Presents itself as an advanced gaming assistant from the future with comprehensive knowledge across all gaming eras

## Setup

### Prerequisites

1. Install the required packages:

```bash
pip install "livekit-agents[openai,silero]~=1.0rc"
```

2. Set up your environment variables:

```bash
# LiveKit credentials
export LIVEKIT_URL=your_livekit_url
export LIVEKIT_API_KEY=your_api_key
export LIVEKIT_API_SECRET=your_api_secret

# OpenAI API key
export OPENAI_API_KEY=your_openai_api_key
```

Alternatively, create a `.env` file in the same directory as the script with these variables.

## Running the Agent

1. Start the agent:

```bash
python gaming_advisor.py dev
```

2. Connect to your agent using:
   - The [Agents Playground](https://agents-playground.livekit.io/)
   - Any app built with LiveKit's client SDKs
   - LiveKit's telephony integration

## How It Works

The Gaming Advisor uses:

- **OpenAI's GPT-4o**: For understanding complex gaming concepts, terminology, and providing expert advice
- **OpenAI's Whisper**: For accurate speech recognition that can handle gaming terms and jargon
- **OpenAI's TTS with Nova voice**: For an energetic, engaging vocal personality that fits gaming context
- **Silero VAD**: For voice activity detection that enables natural conversational flow

## Customization Options

You can customize the NexusGuide agent by:

- Modifying the agent's instructions in the `GamingAdvisorAgent` class to adjust its persona
- Adding new function tools to provide additional capabilities (e.g., esports statistics, game news updates)
- Integrating with game databases APIs (like IGDB, Steam API) for real-time, accurate recommendations
- Adding memory capabilities to remember user preferences across sessions

## Sample Interactions

Users can ask NexusGuide questions like:

- "What games would you recommend for someone who likes strategic RPGs?"
- "I'm stuck on the final boss in Elden Ring. Any tips?"
- "My PC keeps crashing when I try to play Cyberpunk. How can I fix it?"
- "What's a good multiplayer game I can play with friends on different platforms?"
- "How do I optimize my build for a stealth archer in Skyrim?"

## Extension Ideas

- Add image generation capabilities to show game screenshots or visual guides
- Implement streaming integration to demonstrate gameplay while giving advice
- Add multiplayer matchmaking recommendations based on skill level
- Integrate with game news APIs to provide updates on upcoming releases

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.