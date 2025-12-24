import collections
import re
import string
from nltk.tokenize import TreebankWordTokenizer
from sklearn.metrics import accuracy_score, f1_score
import logging

tokenizer = TreebankWordTokenizer()

# Seção: Avaliação de Prompts (Métricas e Extração)
logger = logging.getLogger(__name__)

def extract_label(text: str) -> int | None:
    if not isinstance(text, str): return None
    # Adiciona log para a resposta bruta do LLM antes da extração
    if text and len(text) > 0:
        logger.debug(f"[Extract Label] Resposta bruta do LLM: '{text[:100]}...'")

    match = re.search(r'\b(0|1)\b', text)
    if match: return int(match.group(1))
    return None


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(a_gold, a_pred):
    # Normaliza ambas as strings para garantir uma comparação consistente
    normalized_gold = normalize_text(a_gold)
    normalized_pred = normalize_text(a_pred)
    gold_toks = normalized_gold.split()
    pred_toks = normalized_pred.split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact(a_gold, a_pred):
    return int(normalize_text(a_gold) == normalize_text(a_pred))


def count_tokens(prompt: str) -> int:
    return len(tokenizer.tokenize(prompt))


def calculate_imdb_metrics(true_labels, predictions):
    """Calcula acurácia e F1-score para a tarefa IMDB."""
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary') # Assumindo classificação binária (0 ou 1)
    return acc, f1


def calculate_squad_metrics(total_em, total_f1, dataset_len):
    """Calcula EM e F1-score médios para a tarefa SQuAD."""
    avg_em = total_em / dataset_len
    avg_f1 = total_f1 / dataset_len
    return avg_em, avg_f1