#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: srt_translator_nllb.py
Description: Traduz legendas srt de en-GB para pt-BR.
Author: Thiago Guerreiro
Created on: 2025-11-11
Last modified: 2025-11-12
Version: 1.0.0
License: MIT License
Repository: https://github.com/TGuerreiro/scripts
"""

import pysrt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import sys
from tqdm import tqdm
import re

class TradutorLegendas:
    def __init__(self):
        """Inicializa o tradutor"""
        print("=" * 60)
        print("TRADUTOR DE LEGENDAS")
        print("=" * 60)
        print("\n[1/3] Carregando modelo de tradução...")
        print("(Na primeira vez vai baixar ~6GB, pode demorar!)")
        
        # Usar modelo menor e mais rápido
        modelo = "facebook/nllb-200-1.3B"
        
        print(f"\nModelo: {modelo}")
        print("Aguarde...")
        
        # Instalar safetensors se não tiver
        try:
            import safetensors
        except ImportError:
            print("Instalando safetensors...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors"])
            print("safetensors instalado!")
        
        # Carregar tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        
        # Carregar modelo na GPU
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            modelo,
            torch_dtype=torch.float16,  # Usa menos memória
            use_safetensors=True
        )
        
        # Verificar se CUDA está disponível
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("\nModelo carregado com sucesso!")
            print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memória VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
            self.device = "cuda"
        else:
            print("\n GPU não disponível, usando CPU (será mais lento)")
            self.device = "cpu"
    
    def limpar_texto(self, texto):
        """Remove tags HTML e caracteres especiais problemáticos"""
        # Remover tags HTML comuns em legendas
        texto = re.sub(r'<[^>]+>', '', texto)
        # Normalizar espaços
        texto = re.sub(r'\s+', ' ', texto)
        return texto.strip()
    
    def validar_traducao(self, texto_original, texto_traduzido):
        """Valida se a tradução parece estar em português"""
        # Se o texto traduzido for muito diferente em tamanho, pode haver problema
        if len(texto_traduzido) < 2:
            return False
        
        # Verificar se há caracteres repetidos demais (sinal de erro)
        if re.search(r'(.)\1{8,}', texto_traduzido):
            return False
        
        # Verificar se há caracteres especiais demais (sinal de outro idioma)
        caracteres_especiais = len(re.findall(r'[ąćęłńóśźżđšžčćđ]', texto_traduzido.lower()))
        if caracteres_especiais > len(texto_traduzido) * 0.1:
            return False
        
        return True
    
    def traduzir_texto(self, texto):
        """Traduz um pedaço de texto"""
        # Se estiver vazio, retorna vazio
        if not texto.strip():
            return texto
        
        # Limpar texto
        texto_limpo = self.limpar_texto(texto)
        if not texto_limpo:
            return texto
        
        try:
            # Preparar texto para o modelo
            self.tokenizer.src_lang = "eng_Latn"  # Inglês
            inputs = self.tokenizer(
                texto_limpo, 
                return_tensors="pt", 
                max_length=512,  # Textos maiores
                truncation=True
            ).to(self.device)
            
            tgt_lang = "por_Latn"  # Português em escrita latina
            
            # Obter ID do token para português
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
            else:
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            
            # Traduzir com parâmetros otimizados
            traducao_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,  # Melhor qualidade
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3  # Evita repetições
            )
            
            # Converter tokens de volta para texto
            traducao = self.tokenizer.batch_decode(
                traducao_tokens, 
                skip_special_tokens=True
            )[0]
            
            # Validar tradução
            if not self.validar_traducao(texto_limpo, traducao):
                print(f" Tradução suspeita detectada: '{texto_limpo[:50]}...'")
                # Tentar novamente com parâmetros diferentes
                traducao_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=3,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                traducao = self.tokenizer.batch_decode(
                    traducao_tokens, 
                    skip_special_tokens=True
                )[0]
            
            return traducao
            
        except Exception as erro:
            print(f" Erro ao traduzir: {erro}")
            return texto
    
    def traduzir_lote(self, textos):
        """Traduz múltiplos textos de uma vez (MUITO mais rápido!)"""
        # Filtrar e limpar textos
        textos_limpos = []
        indices_validos = []
        
        for i, t in enumerate(textos):
            texto_limpo = self.limpar_texto(t)
            if texto_limpo.strip():
                textos_limpos.append(texto_limpo)
                indices_validos.append(i)
        
        if not textos_limpos:
            return textos
        
        try:
            # Preparar todos os textos de uma vez
            self.tokenizer.src_lang = "eng_Latn"
            inputs = self.tokenizer(
                textos_limpos,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            tgt_lang = "por_Latn"
            
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
            else:
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            
            # Traduzir tudo de uma vez
            traducao_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decodificar todas as traduções
            traducoes = self.tokenizer.batch_decode(
                traducao_tokens,
                skip_special_tokens=True
            )
            
            # Reconstruir array com traduções nos índices corretos
            resultado = textos.copy()
            for i, idx in enumerate(indices_validos):
                if i < len(traducoes):
                    # Validar cada tradução
                    if self.validar_traducao(textos_limpos[i], traducoes[i]):
                        resultado[idx] = traducoes[i]
                    else:
                        # Se falhar, tentar tradução individual
                        print(f" Retraduzindo item {idx+1}...")
                        resultado[idx] = self.traduzir_texto(textos[idx])
            
            return resultado
            
        except Exception as erro:
            print(f"\n Erro no lote: {erro}")
            print("Processando individualmente...")
            
            # Fallback: processar uma por uma
            resultado = []
            for texto in textos:
                try:
                    resultado.append(self.traduzir_texto(texto))
                except:
                    resultado.append(texto)
            return resultado
    
    def traduzir_arquivo_srt(self, arquivo_entrada, arquivo_saida, tamanho_lote=32):
        """Traduz um arquivo SRT completo"""
        print(f"\n[2/3] Lendo arquivo: {arquivo_entrada}")
        
        # Abrir arquivo de legendas
        legendas = pysrt.open(arquivo_entrada, encoding='utf-8')
        total = len(legendas)
        
        print(f"Arquivo carregado: {total} legendas encontradas")
        print(f"\n[3/3] Traduzindo em lotes de {tamanho_lote}...")
        
        # Processar em lotes
        for i in range(0, total, tamanho_lote):
            lote = legendas[i:i+tamanho_lote]
            textos = [leg.text for leg in lote]
            
            try:
                # Traduzir lote inteiro de uma vez
                traducoes = self.traduzir_lote(textos)
                
                # Atualizar legendas
                for j, legenda in enumerate(lote):
                    if j < len(traducoes):
                        legenda.text = traducoes[j]
                
                # Atualizar progresso
                progresso = min(i + tamanho_lote, total)
                print(f" {progresso}/{total} legendas traduzidas ({(progresso/total)*100:.1f}%)")
                
            except Exception as erro:
                print(f"\n Erro no lote {i//tamanho_lote + 1}: {erro}")
                print("Processando individualmente...")
                
                # Fallback: processar uma por uma
                for legenda in lote:
                    try:
                        legenda.text = self.traduzir_texto(legenda.text)
                    except Exception as e:
                        print(f"Erro ao traduzir: {e}")
        
        # Salvar arquivo traduzido
        print(f"\n Tradução concluída!")
        print(f" Salvando arquivo: {arquivo_saida}")
        legendas.save(arquivo_saida, encoding='utf-8')
        print("\n" + "=" * 60)
        print("SUCESSO! Arquivo traduzido e salvo.")
        print("=" * 60)
    
    def traduzir_pasta(self, pasta_entrada, pasta_saida):
        """Traduz todos os arquivos .srt de uma pasta"""
        # Criar pasta de saída se não existir
        if not os.path.exists(pasta_saida):
            os.makedirs(pasta_saida)
            print(f" Pasta criada: {pasta_saida}")
        
        # Encontrar todos os arquivos .srt
        arquivos_srt = [f for f in os.listdir(pasta_entrada) if f.endswith('.srt')]
        
        if len(arquivos_srt) == 0:
            print(f" ATENÇÃO: Nenhum arquivo .srt encontrado em '{pasta_entrada}'")
            return
        
        print(f"\n Encontrados {len(arquivos_srt)} arquivos .srt")
        print("\nIniciando tradução em lote...\n")
        
        for numero, nome_arquivo in enumerate(arquivos_srt, 1):
            print("\n" + "=" * 60)
            print(f"ARQUIVO {numero}/{len(arquivos_srt)}: {nome_arquivo}")
            print("=" * 60)
            
            caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
            # Adicionar '_pt-BR' no nome do arquivo de saída
            nome_saida = nome_arquivo.replace('.srt', '_pt-BR.srt')
            caminho_saida = os.path.join(pasta_saida, nome_saida)
            
            self.traduzir_arquivo_srt(caminho_entrada, caminho_saida)
        
        print("\n" + "=" * 60)
        print("TODOS OS ARQUIVOS TRADUZIDOS!")
        print("=" * 60)
        print(f"Verifique a pasta: {pasta_saida}")


# ============================================================================
# ÁREA DE CONFIGURAÇÃO - EDITE AQUI!
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                    TRADUTOR DE LEGENDAS                    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Coloque aqui o caminho das suas pastas
    
    PASTA_LEGENDAS_ORIGINAIS = r"C:\projects\translate\TraducaoTopGear\legendas_originais"
    PASTA_LEGENDAS_TRADUZIDAS = r"C:\projects\translate\TraducaoTopGear\legendas_traduzidas"
    
    # Criar o tradutor (vai baixar o modelo na primeira vez)
    tradutor = TradutorLegendas()
    
    # Traduzir TODOS os arquivos da pasta
    tradutor.traduzir_pasta(PASTA_LEGENDAS_ORIGINAIS, PASTA_LEGENDAS_TRADUZIDAS)
    
    print("\n Pressione ENTER para fechar...")
    input()
