import logging
from dataclasses import dataclass, field
from typing import Annotated, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("real-estate-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# Sample voices for different agents
voices = {
    "greeter": "alloy",  # Replace with actual voice IDs from your TTS provider
    "propertyFinder": "echo",
    "viewingScheduler": "alloy",
    "mortgageAdvisor": "onyx",
}

# Sample property database - in a real implementation this would connect to an API
PROPERTY_DATABASE = [
    {
        "id": "P001",
        "address": "123 Main Street",
        "price": 350000,
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft": 1800,
        "type": "Single Family Home",
        "description": "Beautiful single-family home with a spacious backyard, updated kitchen, and hardwood floors throughout.",
    },
    {
        "id": "P002",
        "address": "456 Oak Avenue",
        "price": 275000,
        "bedrooms": 2,
        "bathrooms": 2,
        "sqft": 1200,
        "type": "Condo",
        "description": "Modern condo in the heart of downtown with stunning city views, stainless steel appliances, and a fitness center in the building.",
    },
    {
        "id": "P003",
        "address": "789 Pine Road",
        "price": 425000,
        "bedrooms": 4,
        "bathrooms": 3,
        "sqft": 2400,
        "type": "Single Family Home",
        "description": "Spacious family home in a quiet neighborhood with a two-car garage, finished basement, and newly renovated bathrooms.",
    },
    {
        "id": "P004",
        "address": "101 River Lane",
        "price": 550000,
        "bedrooms": 5,
        "bathrooms": 4,
        "sqft": 3200,
        "type": "Luxury Home",
        "description": "Luxurious home with an open floor plan, gourmet kitchen, master suite with walk-in closet, and a private pool in the backyard.",
    },
]


@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None
    
    property_preferences: dict = field(default_factory=lambda: {
        "min_price": None,
        "max_price": None,
        "min_bedrooms": None,
        "min_bathrooms": None,
        "property_type": None,
        "location": None
    })
    
    viewed_properties: List[str] = field(default_factory=list)
    interested_properties: List[str] = field(default_factory=list)
    
    viewing_date: Optional[str] = None
    viewing_time: Optional[str] = None
    
    prequalified: bool = False
    prequalified_amount: Optional[int] = None
    
    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    
    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "customer_email": self.customer_email or "unknown",
            "property_preferences": self.property_preferences,
            "viewed_properties": self.viewed_properties,
            "interested_properties": self.interested_properties,
            "viewing_scheduled": {
                "date": self.viewing_date,
                "time": self.viewing_time,
            } if self.viewing_date else None,
            "prequalified": {
                "status": self.prequalified,
                "amount": self.prequalified_amount,
            }
        }
        return yaml.dump(data)


RunContext_T = RunContext[UserData]


# Common functions
@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"Thank you, {name}. I've updated your name in our system."


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"Got it. Your phone number ({phone}) has been recorded."


@function_tool()
async def update_email(
    email: Annotated[str, Field(description="The customer's email address")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their email address.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_email = email
    return f"Perfect. I've saved your email address as {email}."


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        llm_model = self.llm or self.session.llm
        if userdata.prev_agent and not isinstance(llm_model, llm.RealtimeModel):
            # only add chat history for non-realtime models for now
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # add instructions including the user data as a system message
        chat_ctx.add_message(
            role="system",
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."

    def _truncate_chat_ctx(
        self,
        items: list[llm.ChatItem],
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list[llm.ChatItem]:
        """Truncate the chat context to keep the last n messages."""

        def _valid_item(item: llm.ChatItem) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in [
                "function_call",
                "function_call_output",
            ]:
                return False
            return True

        new_items: list[llm.ChatItem] = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        # the truncated items should not start with function_call or function_call_output
        while new_items and new_items[0].type in ["function_call", "function_call_output"]:
            new_items.pop(0)

        return new_items


class Greeter(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly virtual real estate agent. Your job is to understand what the caller needs "
                "and direct them to the appropriate specialist on your team. "
                "You can help with property searches, scheduling viewings, or connecting them with a mortgage advisor."
            ),
            llm=openai.LLM(parallel_tool_calls=False),
            tts=openai.TTS(voice=voices["greeter"]),
        )
    
    @function_tool()
    async def to_property_finder(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to search for properties based on their criteria.
        This function handles transitioning to the property finder agent who will collect 
        the necessary details like price range, number of bedrooms, etc."""
        return await self._transfer_to_agent("propertyFinder", context)

    @function_tool()
    async def to_viewing_scheduler(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to schedule a viewing for a property they're interested in.
        This function handles transitioning to the viewing scheduler agent."""
        return await self._transfer_to_agent("viewingScheduler", context)
    
    @function_tool()
    async def to_mortgage_advisor(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to discuss mortgage options or get pre-qualified.
        This function handles transitioning to the mortgage advisor agent."""
        return await self._transfer_to_agent("mortgageAdvisor", context)


class PropertyFinder(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a property finder specialist at a real estate agency. "
                        "Your job is to help customers find properties that match their criteria. "
                        "Ask about their preferences including price range, number of bedrooms and bathrooms, "
                        "property type, and location. Then search for and present matching properties.",
            tools=[update_name, update_phone, update_email],
            tts=openai.TTS(voice=voices["propertyFinder"]),
        )

    @function_tool()
    async def update_property_preferences(
        self,
        min_price: Annotated[Optional[int], Field(description="Minimum price the customer is willing to pay")] = None,
        max_price: Annotated[Optional[int], Field(description="Maximum price the customer is willing to pay")] = None,
        min_bedrooms: Annotated[Optional[int], Field(description="Minimum number of bedrooms required")] = None,
        min_bathrooms: Annotated[Optional[int], Field(description="Minimum number of bathrooms required")] = None,
        property_type: Annotated[Optional[str], Field(description="Type of property (e.g., 'Single Family Home', 'Condo', 'Townhouse')")] = None,
        location: Annotated[Optional[str], Field(description="Preferred location or neighborhood")] = None,
        context: RunContext_T = None,
    ) -> str:
        """Called when the user provides their property preferences."""
        userdata = context.userdata
        
        if min_price:
            userdata.property_preferences["min_price"] = min_price
        if max_price:
            userdata.property_preferences["max_price"] = max_price
        if min_bedrooms:
            userdata.property_preferences["min_bedrooms"] = min_bedrooms
        if min_bathrooms:
            userdata.property_preferences["min_bathrooms"] = min_bathrooms
        if property_type:
            userdata.property_preferences["property_type"] = property_type
        if location:
            userdata.property_preferences["location"] = location
            
        return "I've updated your property preferences. Now I can search for properties that match your criteria."

    @function_tool()
    async def search_properties(
        self,
        context: RunContext_T,
    ) -> str:
        """Called when the user wants to search for properties based on their preferences."""
        userdata = context.userdata
        preferences = userdata.property_preferences
        
        # Filter properties based on preferences (in a real implementation, this would be a database query)
        matching_properties = PROPERTY_DATABASE.copy()
        
        if preferences["min_price"]:
            matching_properties = [p for p in matching_properties if p["price"] >= preferences["min_price"]]
        if preferences["max_price"]:
            matching_properties = [p for p in matching_properties if p["price"] <= preferences["max_price"]]
        if preferences["min_bedrooms"]:
            matching_properties = [p for p in matching_properties if p["bedrooms"] >= preferences["min_bedrooms"]]
        if preferences["min_bathrooms"]:
            matching_properties = [p for p in matching_properties if p["bathrooms"] >= preferences["min_bathrooms"]]
        if preferences["property_type"]:
            matching_properties = [p for p in matching_properties if p["type"].lower() == preferences["property_type"].lower()]
        
        if not matching_properties:
            return "I couldn't find any properties matching your criteria. Would you like to adjust your preferences?"
        
        # Format the results
        result = f"I found {len(matching_properties)} properties matching your criteria:\n\n"
        for i, prop in enumerate(matching_properties, 1):
            result += f"Property {i}: {prop['address']}\n"
            result += f"Price: ${prop['price']:,}\n"
            result += f"{prop['bedrooms']} bed, {prop['bathrooms']} bath, {prop['sqft']} sq ft\n"
            result += f"Type: {prop['type']}\n"
            result += f"Description: {prop['description']}\n\n"
            
            # Add to viewed properties
            if prop["id"] not in userdata.viewed_properties:
                userdata.viewed_properties.append(prop["id"])
        
        return result

    @function_tool()
    async def express_interest(
        self,
        property_address: Annotated[str, Field(description="The address of the property the user is interested in")],
        context: RunContext_T,
    ) -> str:
        """Called when the user expresses interest in a specific property."""
        userdata = context.userdata
        
        # Find the property in the database
        property_found = None
        for prop in PROPERTY_DATABASE:
            if prop["address"].lower() == property_address.lower():
                property_found = prop
                break
        
        if not property_found:
            return f"I couldn't find a property with the address '{property_address}' in our database. Could you please verify the address?"
        
        # Add to interested properties if not already added
        if property_found["id"] not in userdata.interested_properties:
            userdata.interested_properties.append(property_found["id"])
        
        return f"Great! I've noted your interest in the property at {property_address}. Would you like to schedule a viewing or learn more about this property?"

    @function_tool()
    async def to_greeter(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to return to the main menu or speak with another specialist."""
        return await self._transfer_to_agent("greeter", context)

    @function_tool()
    async def to_viewing_scheduler(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to schedule a viewing for a property they're interested in."""
        if not context.userdata.interested_properties:
            return "Before scheduling a viewing, please select at least one property you're interested in."
        return await self._transfer_to_agent("viewingScheduler", context)


class ViewingScheduler(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a viewing scheduler at a real estate agency. "
                        "Your job is to help customers schedule viewings for properties they're interested in. "
                        "First confirm which property they want to view, then collect their preferred date and time, "
                        "and their contact information if we don't already have it.",
            tools=[update_name, update_phone, update_email],
            tts=openai.TTS(voice=voices["viewingScheduler"]),
        )

    @function_tool()
    async def schedule_viewing(
        self,
        property_address: Annotated[str, Field(description="The address of the property to view")],
        date: Annotated[str, Field(description="The preferred date for the viewing (format: YYYY-MM-DD)")],
        time: Annotated[str, Field(description="The preferred time for the viewing (format: HH:MM AM/PM)")],
        context: RunContext_T,
    ) -> str:
        """Called when the user wants to schedule a property viewing."""
        userdata = context.userdata
        
        # Find the property in the database
        property_found = None
        for prop in PROPERTY_DATABASE:
            if prop["address"].lower() == property_address.lower():
                property_found = prop
                break
        
        if not property_found:
            return f"I couldn't find a property with the address '{property_address}' in our database. Could you please verify the address?"
        
        # Make sure we have the customer's contact information
        if not userdata.customer_name or not userdata.customer_phone:
            return "Before I can schedule a viewing, I'll need your name and phone number so our agent can contact you."
        
        # Set the viewing details
        userdata.viewing_date = date
        userdata.viewing_time = time
        
        # In a real implementation, this would create a calendar event or send a notification to a human agent
        
        return f"Great! I've scheduled a viewing for the property at {property_address} on {date} at {time}. One of our agents will meet you there. They may call you at {userdata.customer_phone} to confirm closer to the date."

    @function_tool()
    async def to_greeter(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to return to the main menu or speak with another specialist."""
        return await self._transfer_to_agent("greeter", context)

    @function_tool()
    async def to_mortgage_advisor(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to discuss mortgage options or get pre-qualified."""
        return await self._transfer_to_agent("mortgageAdvisor", context)


class MortgageAdvisor(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a mortgage advisor at a real estate agency. "
                        "Your job is to help customers understand mortgage options and get pre-qualified. "
                        "Ask about their income, credit score, down payment amount, and existing debt "
                        "to determine how much they might qualify for.",
            tools=[update_name, update_phone, update_email],
            tts=openai.TTS(voice=voices["mortgageAdvisor"]),
        )

    @function_tool()
    async def prequalify_mortgage(
        self,
        annual_income: Annotated[int, Field(description="The customer's annual income before taxes")],
        credit_score: Annotated[int, Field(description="The customer's credit score (typically 300-850)")],
        down_payment: Annotated[int, Field(description="The amount the customer can put as a down payment")],
        monthly_debt: Annotated[int, Field(description="The customer's total monthly debt payments (excluding housing)")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides financial information for mortgage pre-qualification."""
        userdata = context.userdata
        
        # Make sure we have the customer's contact information
        if not userdata.customer_name or not userdata.customer_phone or not userdata.customer_email:
            return "Before I can pre-qualify you, I'll need your full contact information (name, phone, and email)."
        
        # Simple pre-qualification logic (in a real implementation, this would be much more complex)
        # Debt-to-income ratio should generally be below 43%
        monthly_income = annual_income / 12
        max_monthly_payment = (monthly_income * 0.43) - monthly_debt
        
        # Rough estimate of loan amount based on a 30-year fixed rate mortgage at 6.5%
        # This is a simplified calculation
        interest_rate = 0.065 / 12  # monthly interest rate
        loan_term_months = 30 * 12  # 30 years in months
        
        # Present value of an annuity formula to calculate maximum loan amount
        loan_amount = max_monthly_payment * ((1 - (1 + interest_rate) ** -loan_term_months) / interest_rate)
        
        # Factor in down payment
        max_home_price = loan_amount + down_payment
        
        # Adjust based on credit score
        if credit_score < 640:
            max_home_price *= 0.8  # Reduce by 20% for poor credit
        elif credit_score < 700:
            max_home_price *= 0.9  # Reduce by 10% for fair credit
        
        # Round to nearest thousand
        max_home_price = round(max_home_price / 1000) * 1000
        
        userdata.prequalified = True
        userdata.prequalified_amount = int(max_home_price)
        
        return f"Based on the information you've provided, I estimate you could qualify for a home up to ${max_home_price:,}. This is just an estimate - a formal pre-approval would require verification of your income, assets, and credit. Would you like me to connect you with a mortgage specialist to get officially pre-approved?"

    @function_tool()
    async def to_greeter(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to return to the main menu or speak with another specialist."""
        return await self._transfer_to_agent("greeter", context)

    @function_tool()
    async def to_property_finder(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to search for properties within their pre-qualified amount."""
        return await self._transfer_to_agent("propertyFinder", context)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(),
            "propertyFinder": PropertyFinder(),
            "viewingScheduler": ViewingScheduler(),
            "mortgageAdvisor": MortgageAdvisor(),
        }
    )
    agent = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="echo"),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await agent.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))