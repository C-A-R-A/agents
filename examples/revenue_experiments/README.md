# Revenue Experiments with LiveKit Agents

This folder contains experiments showing how LiveKit Agents can be used to build AI applications that can generate revenue. Each example demonstrates a different business use case with potential monetization strategies.

## Examples

1. **Virtual Real Estate Agent**: AI agent that helps clients find and view properties, schedule viewings, and qualify leads for real estate agents.

2. **Customer Support Agent**: Multi-agent system for handling customer support inquiries, processing returns, and escalating complex issues to human agents.

## Getting Started

To run these examples, you'll need to:

1. Install the required dependencies:
   ```bash
   pip install "livekit-agents[openai,silero,deepgram,cartesia,turn-detector]~=1.0rc"
   ```

2. Set up the necessary environment variables:
   ```
   LIVEKIT_URL
   LIVEKIT_API_KEY
   LIVEKIT_API_SECRET
   DEEPGRAM_API_KEY
   OPENAI_API_KEY
   ```

3. Run the example:
   ```bash
   python real_estate_agent.py dev
   ```

## Monetization Strategies

These examples demonstrate several potential monetization strategies:

1. **Lead generation fees**: Charge for qualified leads (real estate example)
2. **Transaction fees**: Take a percentage of completed transactions
3. **Subscription model**: Offer the agent as a service with monthly/annual subscriptions
4. **Usage-based pricing**: Charge based on minutes of agent interaction or number of sessions
5. **White-labeling**: Allow businesses to use your agents under their own brand