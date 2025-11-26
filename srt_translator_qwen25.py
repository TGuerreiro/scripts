#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: srt_translator_qwen25.py
Description: Script para tradução das legendas (.srt) da série Top Gear UK de en-GB para pt-BR usando qwen25.
Author: Thiago Guerreiro
Created on: 2025-11-18
Last modified: 2025-11-25
Version: 1.0.0
License: MIT License
Repository: https://github.com/TGuerreiro/scripts
"""

import os
import re
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import List
from dataclasses import dataclass
import requests


class SignalHandler:
    interrupted = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        if not self.interrupted:
            print('\n\nInterrupção detectada. Salvando progresso...')
            self.interrupted = True
        else:
            print('\nForçando encerramento...')
            sys.exit(1)


@dataclass
class SubtitleEntry:
    index: int
    start_time: str
    end_time: str
    text: str
    
    @property
    def timestamp(self) -> str:
        return f'{self.start_time} --> {self.end_time}'
    
    def to_srt(self) -> str:
        return f'{self.index}\n{self.timestamp}\n{self.text}\n'


class TextCleaner:
    
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    SOUND_PATTERNS = [
        re.compile(r'\([A-Z\s]+\)', re.IGNORECASE),
        re.compile(r'\[[A-Z\s]+\]', re.IGNORECASE),
        re.compile(r'♪[^♪]*♪'),
        re.compile(r'#[^#\n]+#'),
    ]
    
    @classmethod
    def clean(cls, text: str) -> str:
        text = cls.HTML_TAG_PATTERN.sub('', text)
        for pattern in cls.SOUND_PATTERNS:
            text = pattern.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class SRTParser:
    
    ENTRY_PATTERN = re.compile(
        r'(\d+)\s*\n\s*'
        r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n'
        r'((?:.*\n?)+?)(?=\n\s*\d+\s*\n|\Z)',
        re.MULTILINE
    )
    
    @classmethod
    def load_file(cls, filepath: str) -> List[SubtitleEntry]:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        entries = []
        for match in cls.ENTRY_PATTERN.finditer(content):
            index = int(match.group(1))
            start_time = match.group(2)
            end_time = match.group(3)
            text = match.group(4).strip()
            text = TextCleaner.clean(text)
            
            if text:
                entries.append(SubtitleEntry(index, start_time, end_time, text))
        
        return entries
    
    @classmethod
    def save_file(cls, entries: List[SubtitleEntry], filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(entries):
                f.write(entry.to_srt())
                if i < len(entries) - 1:
                    f.write('\n')


class OllamaTranslator:
    
    SYSTEM_PROMPT = '''Você é um tradutor profissional especializado em Top Gear UK. Traduza a legenda do inglês britânico para português brasileiro.

ATENÇÃO: Você DEVE responder APENAS em português brasileiro. NUNCA use chinês, japonês, inglês ou qualquer outro idioma. Apenas português do Brasil.

REGRAS OBRIGATÓRIAS:
1. Traduza TUDO para português brasileiro natural e fluente
2. NUNCA use caracteres chineses, japoneses ou de outros idiomas
3. Preserve o humor sarcástico dos apresentadores
4. Use terminologia automotiva brasileira correta
5. Máximo 42 caracteres por linha
6. Seja conciso mas mantenha o sentido completo

TERMINOLOGIA:
- gearbox = câmbio
- bonnet = capô  
- boot = porta-malas
- estate = perua
- saloon = sedã
- windscreen = para-brisa
- petrol = gasolina
- tyre = pneu

CONVERSÕES (faça você mesmo):
- mph = km/h (multiplique por 1.6)
- bhp = cv (aproximado)
- lb-ft = kgfm (divida por 7.2)
- miles = km (multiplique por 1.6)

NÃO TRADUZIR:
- Nomes próprios
- Marcas de carros
- Valores monetários

Responda APENAS com a tradução em português brasileiro, sem explicações, sem notas, sem usar outros idiomas.'''
    
    def __init__(self, model: str, api_url: str):
        self.model = model
        self.api_url = api_url
        self.session = requests.Session()
    
    def has_non_latin_chars(self, text: str) -> bool:
        for char in text:
            if ord(char) > 0x024F and char not in 'áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ':
                return True
        return False
    
    def translate(self, text: str, max_retries: int = 3) -> str:
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': text}
            ],
            'stream': False,
            'options': {
                'temperature': 0.3,
                'top_p': 0.9,
                'num_predict': 200,
                'num_ctx': 4096,
                'num_gpu': 99,
                'num_thread': 8,
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(self.api_url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                translation = result['message']['content'].strip()
                translation = TextCleaner.clean(translation)
                
                if self.has_non_latin_chars(translation):
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return text
                
                return translation
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f'\nErro: {e}')
                    return text
        
        return text


class TranslationEngine:
    
    def __init__(self, translator: OllamaTranslator, signal_handler: SignalHandler):
        self.translator = translator
        self.signal_handler = signal_handler
    
    def translate_file(self, input_path: str, output_path: str):
        entries = SRTParser.load_file(input_path)
        total = len(entries)
        
        print(f'Carregadas {total} legendas\n')
        
        translated = []
        start_time = time.time()
        
        for i, entry in enumerate(entries):
            if self.signal_handler.interrupted:
                print('\nSalvando progresso...')
                break
            
            translated_text = self.translator.translate(entry.text)
            translated.append(SubtitleEntry(
                entry.index,
                entry.start_time,
                entry.end_time,
                translated_text
            ))
            
            if (i + 1) % 10 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                percent = ((i + 1) / total) * 100
                
                bar_length = 40
                filled = int(bar_length * (i + 1) / total)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                
                print(f'\r[{bar}] {i+1}/{total} ({percent:.1f}%) - ETA: {eta_min}m {eta_sec}s - {rate:.1f} leg/s', end='', flush=True)
        
        print('\n')
        
        SRTParser.save_file(translated, output_path)
        
        elapsed = time.time() - start_time
        print(f'Concluído em {int(elapsed//60)}m {int(elapsed%60)}s')
        print(f'Velocidade média: {len(translated)/elapsed:.1f} legendas/segundo')


def check_ollama(api_url: str, model: str) -> bool:
    try:
        base_url = api_url.replace('/api/chat', '')
        response = requests.get(f'{base_url}/api/tags', timeout=5)
        
        if response.status_code != 200:
            print('ERRO: Ollama não está acessível')
            return False
        
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        
        if model not in model_names:
            print(f'ERRO: Modelo "{model}" não encontrado')
            print(f'Modelos disponíveis: {", ".join(model_names)}')
            print(f'\nInstale com: ollama pull {model}')
            return False
        
        return True
    except Exception as e:
        print(f'ERRO: Não foi possível conectar ao Ollama: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Tradutor de legendas Top Gear UK (en-GB para pt-BR)'
    )
    
    parser.add_argument('input', help='Arquivo ou diretório de entrada')
    parser.add_argument('output', help='Arquivo ou diretório de saída')
    parser.add_argument('-m', '--model', default='qwen2.5:32b',
                       help='Modelo Ollama (padrão: qwen2.5:32b)')
    parser.add_argument('--api-url', default='http://localhost:11434/api/chat',
                       help='URL da API Ollama')
    
    args = parser.parse_args()
    
    signal_handler = SignalHandler()
    
    print('Verificando Ollama...')
    if not check_ollama(args.api_url, args.model):
        sys.exit(1)
    
    print(f'Modelo: {args.model}')
    
    translator = OllamaTranslator(args.model, args.api_url)
    engine = TranslationEngine(translator, signal_handler)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        if not input_path.suffix.lower() == '.srt':
            print('ERRO: Arquivo deve ter extensão .srt')
            sys.exit(1)
        
        output_file = output_path if output_path.suffix else output_path / input_path.name
        
        print(f'Traduzindo: {input_path.name}')
        print('=' * 70)
        
        try:
            engine.translate_file(str(input_path), str(output_file))
            print(f'\nSalvo em: {output_file}')
        except KeyboardInterrupt:
            print('\n\nInterrompido.')
            sys.exit(0)
    
    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        srt_files = sorted(input_path.glob('*.srt'))
        
        if not srt_files:
            print(f'ERRO: Nenhum arquivo .srt em {input_path}')
            sys.exit(1)
        
        print(f'Encontrados {len(srt_files)} arquivo(s)\n')
        
        for idx, srt_file in enumerate(srt_files, 1):
            if signal_handler.interrupted:
                break
            
            print('=' * 70)
            print(f'[{idx}/{len(srt_files)}] {srt_file.name}')
            print('=' * 70)
            
            output_file = output_path / srt_file.name
            
            try:
                engine.translate_file(str(srt_file), str(output_file))
                print(f'Salvo em: {output_file}\n')
            except KeyboardInterrupt:
                print('\n\nInterrompido.')
                sys.exit(0)
    
    else:
        print(f'ERRO: Caminho não encontrado: {input_path}')
        sys.exit(1)


if __name__ == '__main__':
    main()
