# ===========================================
# OutputComponent - Simple Output Handler
# ===========================================
"""
Componente simples para output, demonstra configuração e lifecycle básicos.
"""

try:
    from ..main import Component, ComponentConfig
    from ..Events import OutputEvent
except ImportError:
    from main import Component, ComponentConfig
    from Events import OutputEvent
from typing import List
import logging

logger = logging.getLogger(__name__)

class OutputConfig(ComponentConfig):
    """Configuração para OutputComponent"""
    output_format: str = "text"  # "text", "json", "html"
    include_metadata: bool = False
    max_output_length: int = 10000

class OutputComponent(Component):
    """
    Componente simples que processa saídas finais.
    Demonstra configuração básica e lifecycle simples.
    """

    name: str = "OutputComponent"
    config_class = OutputConfig

    # Recebe apenas output events
    receives = [OutputEvent]
    emits = []  # Não emite eventos

    def __init__(self, config: OutputConfig = None, **kwargs):
        super().__init__(config)

        # Callback opcional para custom output handling
        self._output_callback = kwargs.get('output_callback')

    async def on_init(self):
        """Inicializa componente"""
        await super().on_init()
        logger.info("📤 OutputComponent inicializado")

    async def handle_event(self, event: OutputEvent) -> List:  # Retorna vazio
        """Processa evento de output"""
        content = event.content

        # Aplica formatação
        if self.config.output_format == "json":
            import json
            formatted = json.dumps({
                "session_id": event.session_id,
                "content": content,
                "timestamp": event.timestamp
            }, indent=2, ensure_ascii=False)
        elif self.config.output_format == "html":
            formatted = f"<div class='message'>{content}</div>"
        else:
            formatted = content

        # Trunca se necessário
        if len(formatted) > self.config.max_output_length:
            formatted = formatted[:self.config.max_output_length] + "..."

        # Output final
        print(f"\n🤖 Output [{event.session_id}]: {formatted}")

        # Callback customizado se fornecido
        if self._output_callback:
            await self._output_callback(event, formatted)

        # Não emite novos eventos - apenas mostra output
        return []

# Exemplo de output callback personalizado
async def console_output_callback(event: OutputEvent, formatted_content: str):
    """Callback que salva output em arquivo"""
    import os
    os.makedirs("outputs", exist_ok=True)

    with open(f"outputs/session_{event.session_id}.txt", "a", encoding="utf-8") as f:
        f.write(f"[{event.timestamp}] {formatted_content}\n")
        f.write("-" * 50 + "\n")
