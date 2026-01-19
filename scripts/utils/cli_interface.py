# cli_interface.py
"""Interface de linha de comando simplificada para o mo-prompt-project."""
import logging

logger = logging.getLogger(__name__)


def get_validated_input(prompt_message: str, num_options: int) -> int:
    """Obtém input numérico validado do usuário."""
    while True:
        try:
            choice = int(input(prompt_message))
            if 0 <= choice < num_options:
                return choice
            print(f"⚠ Opção inválida. Digite um número entre 0 e {num_options - 1}.")
        except ValueError:
            print("⚠ Entrada inválida. Digite um número.")


def select_from_menu(title: str, options: list, name_key: str = "name") -> tuple[int, dict | str]:
    """
    Exibe um menu de seleção e retorna o índice e item selecionado.
    
    Args:
        title: Título do menu
        options: Lista de opções (dicts ou strings)
        name_key: Chave para extrair nome se options forem dicts
    
    Returns:
        Tuple com (índice, item selecionado)
    """
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)
    
    for i, opt in enumerate(options):
        if isinstance(opt, dict):
            display = opt.get(name_key, f"Opção {i}")
        else:
            display = str(opt)
        print(f"  [{i}] {display}")
    
    print()
    choice = get_validated_input("Sua escolha: ", len(options))
    selected = options[choice]
    
    if isinstance(selected, dict):
        logger.info(f"Selecionado: {selected.get(name_key, selected)}")
    else:
        logger.info(f"Selecionado: {selected}")
    
    return choice, selected


def confirm_action(message: str, default: bool = False) -> bool:
    """Pede confirmação do usuário."""
    suffix = "[S/n]" if default else "[s/N]"
    response = input(f"{message} {suffix}: ").strip().lower()
    
    if not response:
        return default
    return response in ('s', 'sim', 'y', 'yes')


def print_header(text: str, char: str = "=", width: int = 60):
    """Imprime um cabeçalho formatado."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def print_config_summary(config: dict):
    """Exibe um resumo das configurações selecionadas."""
    print_header("Resumo da Configuração", char="-")
    print(f"  • Tarefa: {config.get('task', 'N/A').upper()}")
    print(f"  • Modo: {config.get('objective', 'N/A')}")
    
    evaluator = config.get('evaluators', [{}])[0]
    print(f"  • Avaliador: {evaluator.get('name', 'N/A')}")
    
    strategy = config.get('strategies', [{}])[0]
    print(f"  • Estratégia: {strategy.get('name', 'N/A')}")
    
    print(f"  • Output: {config.get('base_output_dir', 'N/A')}")
    print()
