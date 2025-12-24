import random
from scripts.llm_clients import _call_openai_api
import logging

# Seção: Operadores Evolutivos
logger = logging.getLogger(__name__)

def crossover_and_mutation_ga(pair_of_parent_prompts, config):
    generator_config = config.get("generator")
    if not generator_config:
        logger.error("Configuração do gerador não encontrada.")
        return [{"prompt": "erro_configuracao_gerador"}]
    if len(pair_of_parent_prompts) != 2:
        logger.error("crossover_and_mutation_ga espera exatamente dois pais.")
        return [{"prompt": "erro_numero_pais_invalido"}]
    template_generator = generator_config.get("template_generator", {})
    system_instruction = template_generator.get("system")
    user_instruction_crossover = template_generator.get("user_crossover")
    user_instruction_mutation = template_generator.get("user_mutation", "Mute: {prompt}") # Adicionado valor padrão
    prompt_a = pair_of_parent_prompts[0]["prompt"]
    prompt_b = pair_of_parent_prompts[1]["prompt"]
    crossover_messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_instruction_crossover.format(prompt_a=prompt_a, prompt_b=prompt_b)}]
    crossover_prompt = _call_openai_api(crossover_messages, generator_config)
    if "erro_" in crossover_prompt:
        return [{"prompt": f"prompt_gerado_com_erro_crossover ({crossover_prompt})"}]
    mutation_messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_instruction_mutation.format(prompt=crossover_prompt)}]
    mutated_prompt = _call_openai_api(mutation_messages, generator_config)
    if "erro_" in mutated_prompt:
        return [{"prompt": f"prompt_gerado_com_erro_mutacao ({mutated_prompt})"}]
    else:
        return [{"prompt": mutated_prompt}]


def mop_crossover_and_mutation_ga(pair_of_parent_prompts, config):
    """
    Realiza Crossover e, condicionalmente, Mutação.
    """
    generator_config = config.get("generator")
    
    # Carrega a taxa de mutação do config, default 1.0 (100%)
    evo_params = config.get("evolution_params", {})
    mutation_rate = evo_params.get("mutation_rate", 1.0) 

    if not generator_config:
        return [{"prompt": "erro_configuracao_gerador"}]

    template_generator = generator_config.get("template_generator", {})
    system_instruction = template_generator.get("system", "Você é um otimizador de prompts.")
    user_instruction_crossover = template_generator.get("user_crossover", "Combine: {prompt_a} e {prompt_b}")
    user_instruction_mutation = template_generator.get("rate_user_mutation", "Mute: {prompt}")

    prompt_a = pair_of_parent_prompts[0]["prompt"] if isinstance(pair_of_parent_prompts[0], dict) else pair_of_parent_prompts[0]
    prompt_b = pair_of_parent_prompts[1]["prompt"] if isinstance(pair_of_parent_prompts[1], dict) else pair_of_parent_prompts[1]

    logger.info(f"[Crossover/Mutação] Pais: '{prompt_a[:100]}...' e '{prompt_b[:100]}...'")
    # CROSSOVER
    crossover_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction_crossover.format(prompt_a=prompt_a, prompt_b=prompt_b)}
    ]

    # Temperatura média para o crossover (ex: 0.5 - 0.7)
    current_prompt = _call_openai_api(crossover_messages, generator_config, temperature=0.6)
    logger.info(f"[Crossover/Mutação] Crossover gerado: '{current_prompt[:100]}...'")

    if "erro_" in current_prompt:
        return [{"prompt": f"erro_crossover ({current_prompt})"}]

    # MUTAÇÃO CONDICIONAL
    # Gera um número aleatório entre 0.0 e 1.0
    # Se for MENOR que a taxa (ex: 0.6), aplica a mutação.
    # Caso contrário, o filho é apenas o resultado do crossover.

    if random.random() < mutation_rate:
        # print(f"[evolutionary_operators] Aplicando mutação (Taxa: {mutation_rate})...")
        mutation_messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction_mutation.format(prompt=current_prompt)}
        ]
        # Aumento de temperatura para forçar diversidade se ocorrer a mutação
        logger.info(f"[Crossover/Mutação] Aplicando mutação (Taxa: {mutation_rate})...")
        mutated_prompt = _call_openai_api(mutation_messages, generator_config, temperature=0.9)

        if "erro_" not in mutated_prompt:
            current_prompt = mutated_prompt
        else:
            return [{"prompt": f"erro_mutacao ({mutated_prompt})"}]
    
    return [{"prompt": current_prompt}]