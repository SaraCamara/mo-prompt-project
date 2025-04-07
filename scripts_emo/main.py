import subprocess
import sys
import os

def install_requirements():
    """Instala os pacotes listados no requirements.txt."""
    requirements_file = "requirements.txt"
    
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("[✓] Dependências instaladas com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"[✗] Erro ao instalar dependências: {e}")
    else:
        print(f"[✗] Arquivo {requirements_file} não encontrado.")

def execute_test(modelo, arquivo_csv):
    """Executa o script de inferência com o modelo e o CSV de testes."""
    script_name = "scripts_emo/inferencia.py"

    if not os.path.exists(script_name):
        print(f"[✗] Script {script_name} não encontrado.")
        return
    
    if not os.path.exists(arquivo_csv):
        print(f"[✗] Arquivo de teste {arquivo_csv} não encontrado.")
        return

    subprocess.call([sys.executable, script_name, modelo, arquivo_csv])

if __name__ == "__main__":
    install_requirements()
    execute_test("gemma:2b", "data/imdb_pt_subset.csv")
    #execute_test("maritalk", "data/imdb_pt_subset.csv")