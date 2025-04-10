import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("customer-support-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# Sample voices for different agents
voices = {
    "initial": "alloy",
    "returns": "echo",
    "technical": "nova",
    "billing": "shimmer",
    "manager": "onyx",
}

# Sample product database - in a real implementation this would connect to a database
PRODUCT_DATABASE = [
    {
        "id": "P001",
        "name": "Premium Wireless Headphones",
        "price": 199.99,
        "warranty": "1 year limited warranty",
        "return_period_days": 30,
        "categories": ["Electronics", "Audio"],
    },
    {
        "id": "P002",
        "name": "Ultra HD Smart TV 55\"",
        "price": 699.99,
        "warranty": "2 year limited warranty",
        "return_period_days": 30,
        "categories": ["Electronics", "Television"],
    },
    {
        "id": "P003",
        "name": "Ergonomic Office Chair",
        "price": 249.99,
        "warranty": "5 year limited warranty",
        "return_period_days": 60,
        "categories": ["Furniture", "Office"],
    },
    {
        "id": "P004",
        "name": "Premium Subscription",
        "price": 12.99,
        "billing_cycle": "monthly",
        "categories": ["Services", "Subscription"],
    },
]

# Knowledge base for common technical issues
TECHNICAL_KNOWLEDGE_BASE = {
    "headphones_not_connecting": (
        "If your headphones aren't connecting via Bluetooth: "
        "1. Ensure Bluetooth is enabled on your device. "
        "2. Put the headphones in pairing mode (usually by holding the power button). "
        "3. Make sure the headphones are charged. "
        "4. If previously paired, try removing the device from your Bluetooth settings and reconnect. "
        "5. Reset the headphones by holding the power button for 10 seconds."
    ),
    "tv_no_picture": (
        "If your TV has power but no picture: "
        "1. Check that the correct input source is selected. "
        "2. Verify all cables are securely connected. "
        "3. Try unplugging the TV for 30 seconds, then plug it back in. "
        "4. If using external devices, try disconnecting them and connecting directly to TV. "
        "5. Try a factory reset through your TV settings."
    ),
    "subscription_access_issues": (
        "If you're having trouble accessing your subscription content: "
        "1. Verify your account is active and subscription hasn't expired. "
        "2. Try logging out and back in. "
        "3. Clear your browser cache and cookies. "
        "4. Check if the service is experiencing an outage. "
        "5. Try accessing from a different device or browser."
    ),
}


class IssueType(str, Enum):
    RETURN = "return"
    TECHNICAL = "technical"
    BILLING = "billing"
    OTHER = "other"


@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    
    order_number: Optional[str] = None
    product_id: Optional[str] = None
    issue_type: Optional[IssueType] = None
    issue_description: Optional[str] = None
    
    # Return processing
    return_reason: Optional[str] = None
    return_approved: bool = False
    return_label_sent: bool = False
    
    # Billing
    refund_amount: Optional[float] = None
    refund_approved: bool = False
    
    # Escalation
    escalated: bool = False
    escalation_reason: Optional[str] = None
    
    # Customer satisfaction
    satisfaction_rating: Optional[int] = None
    
    # Agent management
    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    
    def summarize(self) -> str:
        data = {
            "customer_info": {
                "name": self.customer_name or "unknown",
                "email": self.customer_email or "unknown",
                "phone": self.customer_phone or "unknown",
            },
            "issue_details": {
                "order_number": self.order_number or "unknown",
                "product_id": self.product_id or "unknown",
                "issue_type": self.issue_type or "unknown",
                "description": self.issue_description or "unknown",
            },
            "return_status": {
                "reason": self.return_reason,
                "approved": self.return_approved,
                "label_sent": self.return_label_sent,
            } if self.issue_type == IssueType.RETURN else None,
            "billing_status": {
                "refund_amount": self.refund_amount,
                "approved": self.refund_approved,
            } if self.issue_type == IssueType.BILLING else None,
            "escalation": {
                "escalated": self.escalated,
                "reason": self.escalation_reason,
            },
            "satisfaction": self.satisfaction_rating,
        }
        return yaml.dump(data)


RunContext_T = RunContext[UserData]


# Common functions
@function_tool()
async def update_customer_info(
    name: Annotated[Optional[str], Field(description="The customer's full name")] = None,
    email: Annotated[Optional[str], Field(description="The customer's email address")] = None,
    phone: Annotated[Optional[str], Field(description="The customer's phone number")] = None,
    context: RunContext_T = None,
) -> str:
    """Called when the user provides their contact information.
    Confirm the information with the user before calling the function."""
    userdata = context.userdata
    
    if name:
        userdata.customer_name = name
    if email:
        userdata.customer_email = email
    if phone:
        userdata.customer_phone = phone
    
    updated_fields = []
    if name:
        updated_fields.append("name")
    if email:
        updated_fields.append("email")
    if phone:
        updated_fields.append("phone")
    
    return f"Thank you, I've updated your {', '.join(updated_fields)}."


@function_tool()
async def record_satisfaction(
    rating: Annotated[int, Field(description="Customer satisfaction rating on a scale of 1-5", ge=1, le=5)],
    context: RunContext_T,
) -> str:
    """Called when the customer provides a satisfaction rating for the support experience."""
    userdata = context.userdata
    userdata.satisfaction_rating = rating
    
    if rating >= 4:
        return "Thank you for your positive feedback! We're glad we could help you today."
    elif rating == 3:
        return "Thank you for your feedback. We're always working to improve our service."
    else:
        return "I'm sorry to hear that. We take your feedback seriously and will use it to improve our service."


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


class InitialSupport(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the initial customer support agent for an electronics and home goods company. "
                "Your job is to greet the customer, identify their issue, and route them to the appropriate "
                "specialized agent. Be friendly and efficient in collecting the basic information needed."
            ),
            llm=openai.LLM(parallel_tool_calls=False),
            tts=openai.TTS(voice=voices["initial"]),
        )

    @function_tool()
    async def identify_issue(
        self,
        order_number: Annotated[Optional[str], Field(description="The customer's order number if applicable")] = None,
        product_id: Annotated[Optional[str], Field(description="The product ID if applicable")] = None,
        issue_type: Annotated[IssueType, Field(description="The type of issue the customer is experiencing")],
        description: Annotated[str, Field(description="Brief description of the customer's issue")],
        context: RunContext_T,
    ) -> str:
        """Called when the agent has identified the customer's issue type and basic details."""
        userdata = context.userdata
        
        if order_number:
            userdata.order_number = order_number
        if product_id:
            userdata.product_id = product_id
        userdata.issue_type = issue_type
        userdata.issue_description = description
        
        return f"Thank you for providing those details. I understand you're having a {issue_type} issue. I'll route you to the appropriate specialist."
    
    @function_tool()
    async def to_returns(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer has a return or refund request."""
        userdata = context.userdata
        if not userdata.issue_type:
            userdata.issue_type = IssueType.RETURN
        return await self._transfer_to_agent("returns", context)
    
    @function_tool()
    async def to_technical(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer has a technical issue with a product."""
        userdata = context.userdata
        if not userdata.issue_type:
            userdata.issue_type = IssueType.TECHNICAL
        return await self._transfer_to_agent("technical", context)
    
    @function_tool()
    async def to_billing(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer has a billing or payment issue."""
        userdata = context.userdata
        if not userdata.issue_type:
            userdata.issue_type = IssueType.BILLING
        return await self._transfer_to_agent("billing", context)
    
    @function_tool()
    async def to_manager(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer needs to speak with a manager or has a complex issue."""
        userdata = context.userdata
        userdata.escalated = True
        userdata.escalation_reason = "Customer requested manager"
        return await self._transfer_to_agent("manager", context)


class ReturnsAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a returns specialist for an electronics and home goods company. "
                "Your job is to help customers process returns and refunds. Collect the necessary "
                "information about the return reason, verify eligibility, and process the return "
                "if applicable. Be empathetic but follow company policies."
            ),
            tools=[update_customer_info, record_satisfaction],
            tts=openai.TTS(voice=voices["returns"]),
        )
    
    @function_tool()
    async def process_return(
        self,
        return_reason: Annotated[str, Field(description="The reason for the return")],
        context: RunContext_T,
    ) -> str:
        """Called when the customer provides the reason for returning a product."""
        userdata = context.userdata
        
        if not userdata.order_number or not userdata.product_id:
            return "Before I can process your return, I'll need your order number and the product ID. Do you have those available?"
        
        # In a real implementation, this would check a database for order validity and return eligibility
        # For this example, we'll assume all products are eligible for return if they exist
        product_found = None
        for product in PRODUCT_DATABASE:
            if product["id"] == userdata.product_id:
                product_found = product
                break
        
        if not product_found:
            return f"I'm unable to find product ID {userdata.product_id} in our system. Could you please verify the product ID?"
        
        # Check if it's a subscription, which has different return policies
        if "Subscription" in product_found.get("categories", []):
            return "This appears to be a subscription service, which follows our digital services cancellation policy. Let me transfer you to our billing department who can help with cancellations and refunds."
        
        userdata.return_reason = return_reason
        userdata.return_approved = True
        
        return "Thank you for providing that information. Based on our policy, your return has been approved. Would you like me to email you a return shipping label?"
    
    @function_tool()
    async def send_return_label(
        self,
        context: RunContext_T,
    ) -> str:
        """Called when the customer confirms they want a return shipping label sent to them."""
        userdata = context.userdata
        
        if not userdata.return_approved:
            return "I see that your return hasn't been approved yet. Let's first verify if your product is eligible for return."
        
        if not userdata.customer_email:
            return "I'll need your email address to send the return label. Could you please provide that?"
        
        userdata.return_label_sent = True
        
        # In a real implementation, this would trigger an email with a return label
        return f"Great! I've sent a return shipping label to {userdata.customer_email}. Once you ship the item back, your refund will be processed within 5-7 business days after we receive it. Is there anything else I can help you with today?"
    
    @function_tool()
    async def to_initial(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer has a different issue or wants to start over."""
        return await self._transfer_to_agent("initial", context)
    
    @function_tool()
    async def to_billing(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the return issue involves billing or refunds that need special handling."""
        return await self._transfer_to_agent("billing", context)
    
    @function_tool()
    async def to_manager(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer is unsatisfied with the return policy or has a complex issue."""
        userdata = context.userdata
        userdata.escalated = True
        userdata.escalation_reason = "Complex return issue"
        return await self._transfer_to_agent("manager", context)


class TechnicalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a technical support specialist for an electronics and home goods company. "
                "Your job is to help customers troubleshoot and resolve technical issues with their products. "
                "Provide clear, step-by-step instructions and verify if the suggested solutions work."
            ),
            tools=[update_customer_info, record_satisfaction],
            tts=openai.TTS(voice=voices["technical"]),
        )
    
    @function_tool()
    async def troubleshoot_issue(
        self,
        context: RunContext_T,
    ) -> str:
        """Called to access the knowledge base and provide troubleshooting steps for the customer's issue."""
        userdata = context.userdata
        
        if not userdata.product_id and not userdata.issue_description:
            return "To help troubleshoot your issue, I'll need more details about the product and the problem you're experiencing. Could you describe what's happening?"
        
        # Search the knowledge base for relevant troubleshooting info
        # In a real implementation, this would use a more sophisticated search
        issue_description = userdata.issue_description.lower() if userdata.issue_description else ""
        
        if "headphone" in issue_description and ("connect" in issue_description or "bluetooth" in issue_description or "pair" in issue_description):
            return TECHNICAL_KNOWLEDGE_BASE["headphones_not_connecting"]
        elif "tv" in issue_description and ("picture" in issue_description or "display" in issue_description or "screen" in issue_description):
            return TECHNICAL_KNOWLEDGE_BASE["tv_no_picture"]
        elif "subscription" in issue_description and ("access" in issue_description or "login" in issue_description or "content" in issue_description):
            return TECHNICAL_KNOWLEDGE_BASE["subscription_access_issues"]
        else:
            # Generic response for issues not in the knowledge base
            return (
                "Based on the information you've provided, I recommend the following general troubleshooting steps:\n\n"
                "1. Power cycle the device (turn it off, unplug it for 30 seconds, then plug it back in and turn it on).\n"
                "2. Ensure all connections are secure and properly attached.\n"
                "3. Check for any available software or firmware updates.\n"
                "4. Try using the device in a different environment or setup if possible.\n\n"
                "Did any of these steps help resolve your issue?"
            )
    
    @function_tool()
    async def escalate_technical_issue(
        self,
        reason: Annotated[str, Field(description="The reason for escalating the technical issue")],
        context: RunContext_T,
    ) -> tuple[Agent, str]:
        """Called when the technical issue can't be resolved with basic troubleshooting."""
        userdata = context.userdata
        userdata.escalated = True
        userdata.escalation_reason = reason
        
        # In a real implementation, this might create a support ticket
        
        return await self._transfer_to_agent("manager", context)
    
    @function_tool()
    async def to_initial(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer has a different issue or wants to start over."""
        return await self._transfer_to_agent("initial", context)
    
    @function_tool()
    async def to_returns(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer wants to return a product after troubleshooting."""
        return await self._transfer_to_agent("returns", context)


class BillingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a billing and payments specialist for an electronics and home goods company. "
                "Your job is to help customers with billing inquiries, process refunds, manage subscriptions, "
                "and resolve payment issues. Be precise and trustworthy when handling financial matters."
            ),
            tools=[update_customer_info, record_satisfaction],
            tts=openai.TTS(voice=voices["billing"]),
        )
    
    @function_tool()
    async def process_refund(
        self,
        amount: Annotated[float, Field(description="The amount to be refunded")],
        context: RunContext_T,
    ) -> str:
        """Called when a refund needs to be processed for the customer."""
        userdata = context.userdata
        
        if not userdata.order_number:
            return "Before I can process a refund, I'll need your order number. Do you have that available?"
        
        if not userdata.customer_email:
            return "I'll need your email address to process the refund. Could you please provide that?"
        
        # In a real implementation, this would verify the order and eligible refund amount
        userdata.refund_amount = amount
        userdata.refund_approved = True
        
        # In a real implementation, this would trigger the refund process
        return f"I've processed a refund of ${amount:.2f} for your order. The refund will be credited back to your original payment method within 5-7 business days. You'll receive a confirmation email at {userdata.customer_email}. Is there anything else I can help you with today?"
    
    @function_tool()
    async def manage_subscription(
        self,
        action: Annotated[str, Field(description="The action to take on the subscription (cancel, pause, resume)")],
        context: RunContext_T,
    ) -> str:
        """Called when the customer wants to manage their subscription service."""
        userdata = context.userdata
        
        if not userdata.customer_email:
            return "I'll need your email address to locate your subscription. Could you please provide that?"
        
        # In a real implementation, this would verify the subscription and perform the requested action
        
        if action.lower() == "cancel":
            return f"I've cancelled your subscription. You'll have access until the end of your current billing period. You'll receive a confirmation email at {userdata.customer_email}. Is there anything else I can help you with today?"
        elif action.lower() == "pause":
            return f"I've paused your subscription for 30 days. Your billing will resume after that period. You'll receive a confirmation email at {userdata.customer_email}. Is there anything else I can help you with today?"
        elif action.lower() == "resume":
            return f"I've resumed your subscription. Your next billing date will be updated accordingly. You'll receive a confirmation email at {userdata.customer_email}. Is there anything else I can help you with today?"
        else:
            return f"I'm not sure what action you want to take on your subscription. Would you like to cancel, pause, or resume your subscription?"
    
    @function_tool()
    async def to_initial(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the customer has a different issue or wants to start over."""
        return await self._transfer_to_agent("initial", context)
    
    @function_tool()
    async def to_manager(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the billing issue is complex or requires manager approval."""
        userdata = context.userdata
        userdata.escalated = True
        userdata.escalation_reason = "Complex billing issue"
        return await self._transfer_to_agent("manager", context)


class ManagerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a customer support manager with authority to handle escalated issues and exceptions. "
                "Your job is to resolve complex problems, address customer dissatisfaction, and make policy exceptions "
                "when appropriate. Balance customer satisfaction with company policies and be empowered to offer "
                "special accommodations in reasonable situations."
            ),
            tools=[update_customer_info, record_satisfaction],
            tts=openai.TTS(voice=voices["manager"]),
        )
    
    @function_tool()
    async def resolve_escalated_issue(
        self,
        resolution: Annotated[str, Field(description="The resolution offered to the customer")],
        special_accommodation: Annotated[Optional[str], Field(description="Any special accommodation or exception made")] = None,
        context: RunContext_T,
    ) -> str:
        """Called when the manager has determined a resolution for the escalated issue."""
        userdata = context.userdata
        
        if not userdata.escalated:
            return "I'm not seeing any escalated issue in our system. Could you please explain the issue you're experiencing?"
        
        response = f"I understand this has been a frustrating experience, and I appreciate your patience. Here's what I can do to resolve this issue: {resolution}"
        
        if special_accommodation:
            response += f" Additionally, as a one-time special accommodation, I'm also offering: {special_accommodation}"
        
        response += " Is this resolution satisfactory for you?"
        
        return response
    
    @function_tool()
    async def to_initial(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the issue is resolved or the customer has a new issue."""
        return await self._transfer_to_agent("initial", context)
    
    @function_tool()
    async def to_returns(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the manager determines the issue should be handled by the returns department."""
        return await self._transfer_to_agent("returns", context)
    
    @function_tool()
    async def to_technical(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the manager determines the issue should be handled by technical support."""
        return await self._transfer_to_agent("technical", context)
    
    @function_tool()
    async def to_billing(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the manager determines the issue should be handled by the billing department."""
        return await self._transfer_to_agent("billing", context)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = UserData()
    userdata.agents.update(
        {
            "initial": InitialSupport(),
            "returns": ReturnsAgent(),
            "technical": TechnicalAgent(),
            "billing": BillingAgent(),
            "manager": ManagerAgent(),
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
        agent=userdata.agents["initial"],
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))