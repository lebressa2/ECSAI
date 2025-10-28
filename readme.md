# 🧠 ECSAI Framework

> **Build Modular AI Agents with the Entity-Component-System Pattern**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Private-red)](#)
[![Framework](https://img.shields.io/badge/framework-ECS-ff69b4)](#)
[![Type Safe](https://img.shields.io/badge/type--safe-pydantic-green)](#)

ECSAI is a **low-level framework** for building scalable, modular AI agents using the Entity-Component-System (ECS) architectural pattern. Like [LangGraph](https://github.com/langchain-ai/langgraph), we provide the **primitives** — you build the intelligence.

```python
from ecsaai import Component, BaseEvent, Agent

class LLMRequest(BaseEvent):
    type: str = "llm_request"
    prompt: str

class MyLLMComponent(Component):
    name = "llm"
    receives = [LLMRequest]
    emits = [LLMResponse]
    
    async def handle_event(self, event):
        response = await self.llm.ainvoke(event.prompt)
        return [LLMResponse(...)]

agent = Agent("my_agent")
await agent.add_component(MyLLMComponent())
```

---

## 🎯 Why ECSAI?

### Traditional AI Agent Frameworks
```python
# Tightly coupled, hard to test, opinionated
class ChatBot:
    def __init__(self):
        self.llm = OpenAI()
        self.memory = Memory()
        self.tools = [tool1, tool2]
    
    def chat(self, message):
        context = self.memory.get()
        response = self.llm.chat(message, context)
        self.memory.save(response)
        return response
```

### ECSAI Approach
```python
# Modular, testable, composable
class ContextComponent(Component):
    receives = [UserMessage]
    emits = [LLMRequest]
    
class LLMComponent(Component):
    receives = [LLMRequest]
    emits = [LLMResponse]

class OutputComponent(Component):
    receives = [LLMResponse]
    emits = [OutputEvent]

# Mix and match components like LEGO blocks
agent = Agent("chatbot")
agent.add_component(ContextComponent())
agent.add_component(LLMComponent())
agent.add_component(OutputComponent())
```

**Key Benefits:**
- 🧩 **Modular** - Each component is independent and reusable
- 🔄 **Event-Driven** - Type-safe communication via events
- 🧪 **Testable** - Test components in isolation
- 📊 **Observable** - Built-in metrics and logging
- 🔌 **Extensible** - Add new components without breaking existing ones

---

## 📦 Installation

```bash
# From source
git clone https://github.com/lebressa2/ECSAI.git
cd ECSAI
pip install -e .

# Direct from GitHub
pip install git+https://github.com/lebressa2/ECSAI.git
```

**Requirements:** Python 3.10+

---

## 🚀 Quick Start

### 1️⃣ Define Your Events

Events are type-safe messages that components exchange:

```python
from ecsaai import BaseEvent
from pydantic import Field

class UserMessage(BaseEvent):
    """User input event"""
    type: str = "user_message"
    content: str
    user_id: str
    session_id: str

class BotResponse(BaseEvent):
    """Bot output event"""
    type: str = "bot_response"
    content: str
    metadata: dict = Field(default_factory=dict)
```

### 2️⃣ Create Your Component

Components process events and emit new ones:

```python
from ecsaai import Component
from typing import List

class EchoComponent(Component):
    """Simple echo component for demonstration"""
    
    name = "echo"
    receives = [UserMessage]  # What events it handles
    emits = [BotResponse]     # What events it produces
    
    async def on_init(self):
        """Called when component initializes"""
        self.echo_count = 0
        print(f"✅ {self.name} component initialized")
    
    async def handle_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Process incoming events"""
        if isinstance(event, UserMessage):
            self.echo_count += 1
            
            return [BotResponse(
                sender=self.name,
                target=event.sender,
                content=f"Echo #{self.echo_count}: {event.content}",
                metadata={"original_user": event.user_id}
            )]
        
        return []
    
    async def on_shutdown(self):
        """Cleanup when component stops"""
        print(f"🛑 {self.name} processed {self.echo_count} messages")
```

### 3️⃣ Wire It Up

Create an agent and add your components:

```python
import asyncio
from ecsaai import Agent

async def main():
    # Create agent
    agent = Agent("echo_bot")
    
    # Add component
    await agent.add_component(EchoComponent())
    
    # Initialize
    await agent.init_all()
    
    # Send event
    results = await agent.send_event(
        UserMessage(
            sender="user_123",
            target="echo_bot:echo",
            content="Hello, ECSAI!",
            user_id="user_123",
            session_id="session_456"
        )
    )
    
    # Process results
    for event in results:
        if isinstance(event, BotResponse):
            print(f"🤖 Bot: {event.content}")
    
    # Cleanup
    await agent.shutdown_all()

asyncio.run(main())
```

**Output:**
```
✅ echo component initialized
🤖 Bot: Echo #1: Hello, ECSAI!
🛑 echo processed 1 messages
```

---

## 🏗️ Core Concepts

### Events: Type-Safe Messages

```python
from ecsaai import BaseEvent
from pydantic import Field

class CustomEvent(BaseEvent):
    type: str = "custom"
    
    # Your custom fields with validation
    user_input: str = Field(..., min_length=1)
    priority: int = Field(default=1, ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    
    # Built-in fields (inherited)
    # - event_id: str (auto-generated UUID)
    # - sender: str (who sent it)
    # - target: str (who should receive it)
    # - timestamp: float (when it was created)
    # - meta: dict (arbitrary metadata)
```

### Components: Independent Processors

```python
from ecsaai import Component, BaseEvent, ComponentConfig
from typing import List

class MyComponentConfig(ComponentConfig):
    """Optional: typed configuration"""
    api_key: str
    max_retries: int = 3

class MyComponent(Component):
    name = "my_component"
    receives = [InputEvent]
    emits = [OutputEvent, ErrorEvent]
    config_class = MyComponentConfig  # Optional
    
    async def on_init(self):
        """Lifecycle: initialization"""
        self.client = APIClient(self.config.api_key)
    
    async def handle_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Core: event processing"""
        if isinstance(event, InputEvent):
            result = await self.process(event)
            return [OutputEvent(...)]
        return []
    
    async def on_error(self, event: BaseEvent, error: Exception):
        """Lifecycle: error handling"""
        return [ErrorEvent(error=str(error))]
    
    async def on_shutdown(self):
        """Lifecycle: cleanup"""
        await self.client.close()
```

### Agents: Event Orchestrators

```python
from ecsaai import Agent

agent = Agent("my_agent")

# Add components
await agent.add_component(Component1())
await agent.add_component(Component2())

# Initialize all
await agent.init_all()

# Send events (external entry point)
results = await agent.send_event(MyEvent(...))

# Internal event dispatch
internal_results = await agent.dispatch_intra([Event1(), Event2()])

# Cleanup
await agent.shutdown_all()
```

---

## 🎨 Realistic Examples

### Example 1: LLM-Powered Chatbot

```python
from ecsaai import Component, BaseEvent
from langchain_openai import ChatOpenAI
from typing import List

# Events
class UserMessage(BaseEvent):
    type: str = "user_message"
    content: str

class LLMRequest(BaseEvent):
    type: str = "llm_request"
    prompt: str
    context: list = []

class LLMResponse(BaseEvent):
    type: str = "llm_response"
    content: str

# Components
class ContextComponent(Component):
    """Manages conversation history"""
    name = "context"
    receives = [UserMessage]
    emits = [LLMRequest]
    
    async def on_init(self):
        self.history = []
    
    async def handle_event(self, event):
        if isinstance(event, UserMessage):
            self.history.append({"role": "user", "content": event.content})
            
            return [LLMRequest(
                sender=self.name,
                target=f"{self.agent_id}:llm",
                prompt=event.content,
                context=self.history[-5:]  # Last 5 messages
            )]
        return []

class LLMComponent(Component):
    """Calls OpenAI API"""
    name = "llm"
    receives = [LLMRequest]
    emits = [LLMResponse]
    
    async def on_init(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    async def handle_event(self, event):
        if isinstance(event, LLMRequest):
            messages = event.context + [
                {"role": "user", "content": event.prompt}
            ]
            
            response = await self.llm.ainvoke(messages)
            
            return [LLMResponse(
                sender=self.name,
                target=event.sender,
                content=response.content
            )]
        return []

# Usage
agent = Agent("chatbot")
await agent.add_component(ContextComponent())
await agent.add_component(LLMComponent())
await agent.init_all()

results = await agent.send_event(UserMessage(
    sender="user",
    target="chatbot:context",
    content="What is the capital of France?"
))
```

### Example 2: Multi-Step RAG Pipeline

```python
class QueryComponent(Component):
    """Receives user query"""
    receives = [UserQuery]
    emits = [SearchRequest]

class VectorSearchComponent(Component):
    """Searches vector database"""
    receives = [SearchRequest]
    emits = [SearchResults]
    
    async def on_init(self):
        self.vectordb = ChromaDB()

class SynthesisComponent(Component):
    """Synthesizes answer from results"""
    receives = [SearchResults]
    emits = [LLMRequest]

class LLMComponent(Component):
    """Generates final response"""
    receives = [LLMRequest]
    emits = [FinalResponse]

# Chain them together
agent = Agent("rag_system")
await agent.add_component(QueryComponent())
await agent.add_component(VectorSearchComponent())
await agent.add_component(SynthesisComponent())
await agent.add_component(LLMComponent())
```

### Example 3: Multi-Agent System

```python
from ecsaai import EventBus

# Create specialized agents
research_agent = Agent("researcher")
await research_agent.add_component(WebSearchComponent())
await research_agent.add_component(SummarizerComponent())

writing_agent = Agent("writer")
await writing_agent.add_component(ContentGeneratorComponent())
await writing_agent.add_component(EditorComponent())

# Coordinate with event bus
bus = EventBus()
bus.register_agent("researcher", research_agent)
bus.register_agent("writer", writing_agent)

# Research agent sends to writer agent
results = await bus.dispatch([
    ResearchRequest(
        sender="user",
        target="researcher:search",
        query="AI trends 2025"
    )
])

# Automatically routes to writer when research completes
```

---

## 📊 Built-in Observability

Every component automatically tracks metrics:

```python
# Get component metrics
metrics = component.metrics

print(f"Events received: {metrics.events_received}")
print(f"Events emitted: {metrics.events_emitted}")
print(f"Errors: {metrics.errors}")
print(f"Avg latency: {metrics.avg_latency_ms:.2f}ms")
print(f"Last error: {metrics.last_error}")
```

**Output:**
```
Events received: 1523
Events emitted: 1523
Errors: 3
Avg latency: 45.32ms
Last error: RateLimitError: Too many requests
```

---

## 🧪 Testing Components

Components are designed to be easily testable:

```python
import pytest
from ecsaai import ComponentTestHarness

@pytest.mark.asyncio
async def test_echo_component():
    # Setup
    component = EchoComponent()
    harness = ComponentTestHarness(component)
    await component.on_init()
    
    # Send event
    results = await harness.send(UserMessage(
        sender="test",
        target="echo",
        content="test message",
        user_id="test_user",
        session_id="test_session"
    ))
    
    # Assert
    assert len(results) == 1
    assert isinstance(results[0], BotResponse)
    assert "test message" in results[0].content
    
    # Check metrics
    metrics = harness.get_metrics()
    assert metrics.events_received == 1
    assert metrics.events_emitted == 1
    assert metrics.errors == 0
```

---

## 🔌 Advanced Features

### Middleware

Add cross-cutting concerns without modifying components:

```python
from ecsaai import ComponentMiddleware

class LoggingMiddleware(ComponentMiddleware):
    async def before_handle(self, event):
        print(f"📥 Received: {event.type}")
        return event
    
    async def after_handle(self, event, results):
        print(f"📤 Emitted: {len(results)} events")
        return results

class RateLimitMiddleware(ComponentMiddleware):
    def __init__(self, max_per_minute=60):
        self.rate_limiter = RateLimiter(max_per_minute)
    
    async def before_handle(self, event):
        if not self.rate_limiter.allow():
            raise RateLimitError("Too many requests")
        return event

# Apply to component
component.add_middleware(LoggingMiddleware())
component.add_middleware(RateLimitMiddleware(max_per_minute=100))
```

### Component Configuration

Type-safe configuration with Pydantic:

```python
from ecsaai import ComponentConfig
from pydantic import Field, validator

class LLMConfig(ComponentConfig):
    provider: str = Field(..., description="LLM provider")
    model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    api_key: str = Field(..., description="API key")
    
    @validator("provider")
    def validate_provider(cls, v):
        allowed = ["openai", "anthropic", "google"]
        if v not in allowed:
            raise ValueError(f"Provider must be one of {allowed}")
        return v

class LLMComponent(Component):
    config_class = LLMConfig
    
    async def on_init(self):
        self.llm = create_llm(
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
            api_key=self.config.api_key
        )

# Usage with validation
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="sk-..."
)
component = LLMComponent(config=config)
```

### Error Handling

Automatic error handling with custom recovery:

```python
class RobustComponent(Component):
    async def on_error(self, event, error):
        """Custom error handling"""
        
        if isinstance(error, RateLimitError):
            # Retry after delay
            await asyncio.sleep(60)
            return await self.handle_event(event)
        
        elif isinstance(error, ValidationError):
            # Return error event
            return [ErrorEvent(
                sender=self.name,
                target=event.sender,
                error=f"Invalid input: {error}"
            )]
        
        else:
            # Default error handling
            return await super().on_error(event, error)
```

---

## 🆚 Comparison with Other Frameworks

| Feature | ECSAI | LangGraph | CrewAI | AutoGen |
|---------|-------|-----------|--------|---------|
| **Architecture** | ECS Pattern | Graph-based | Agent Roles | Multi-Agent Chat |
| **Flexibility** | ✅ High | ✅ High | ⚠️ Medium | ⚠️ Medium |
| **Type Safety** | ✅ Pydantic | ✅ TypedDict | ❌ | ❌ |
| **Learning Curve** | 📚 Medium | 📚 Medium | 📕 Easy | 📚 Medium |
| **Built-in Observability** | ✅ Yes | ⚠️ Limited | ⚠️ Limited | ❌ No |
| **Component Isolation** | ✅ Full | ⚠️ Partial | ❌ Coupled | ❌ Coupled |
| **Testing Support** | ✅ Excellent | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| **Use Case** | Modular Agents | Workflows | Team Simulation | Conversations |

**Choose ECSAI when:**
- ✅ You need truly modular, reusable components
- ✅ You want type-safe event communication
- ✅ You value testability and observability
- ✅ You're building complex, scalable agent systems

---

## 📚 Documentation

- **[Philosophy & Architecture](docs/philosophy.md)** - Why ECS for AI agents
- **[Core Concepts](docs/core-concepts.md)** - Events, Components, Agents
- **[Component Development Guide](docs/component-guide.md)** - Building components
- **[API Reference](docs/api-reference.md)** - Complete API docs
- **[Examples](examples/)** - Progressive examples
- **[Best Practices](docs/best-practices.md)** - Patterns & anti-patterns

---

## 🗺️ Roadmap

- [ ] **v0.2.0** - Event streaming & async iterators
- [ ] **v0.3.0** - OpenTelemetry integration
- [ ] **v0.4.0** - Component marketplace
- [ ] **v0.5.0** - Visual workflow editor
- [ ] **v1.0.0** - Production-ready stable release

---

## 🤝 Contributing

We welcome contributions! Since this is a **low-level framework**, focus on:

- 🧩 **Core primitives** improvements
- 📊 **Observability** enhancements
- 🧪 **Testing utilities**
- 📚 **Documentation & examples**
- 🔌 **Developer experience** tools

**Not looking for:**
- ❌ Pre-built components (those belong in examples/)
- ❌ Opinionated workflows
- ❌ Framework lock-in features

---

## 📄 License

**Private License** - All rights reserved © 2025

---

## 💬 Support & Community

- 📧 **Email:** [lebressanin@gmail.com](mailto:lebressanin@gmail.com)
- 📞 **WhatsApp:** +55 14 99183-5600
- 🐛 **Issues:** [GitHub Issues](https://github.com/lebressa2/ECSAI/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/lebressa2/ECSAI/discussions)

---

<div align="center">

**Built with ❤️ for developers who value modularity and type safety**

⭐ **Star this repo** if you find it useful!

[Get Started](#-quick-start) • [Documentation](#-documentation) • [Examples](examples/)

</div>
