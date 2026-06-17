# Sistema de Controle de Posição de Junta Robótica com Rejeição de Distúrbios

Projeto de controle automático desenvolvido para uma junta robótica acionada por motor DC, utilizando um controlador PI digital para rastreamento de posição e rejeição de perturbações externas.

## Objetivos

- Controlar a posição angular de uma junta robótica.
- Garantir erro em regime permanente próximo de zero.
- Rejeitar distúrbios de carga aplicados ao sistema.
- Atender requisitos de desempenho e estabilidade.
- Disponibilizar uma interface gráfica para análise em tempo real.

---

# Estrutura do Projeto

```
.
├── Relatorio_Controle_Junta_Robotica.docx
├── simulacao_completa.py
├── analise_offline.py
├── 01_Resposta_Degrau.png
├── 02_Rejeicao_Disturbio.png
├── Relatorio_Validacao.txt
└── README.md
```

---

# Arquivos

## Relatório Técnico

**Relatorio_Controle_Junta_Robotica.docx**

Contém:

- Modelagem matemática da planta
- Projeto do controlador PI
- Análise de estabilidade
- Lugar das Raízes
- Diagramas de Bode e Nyquist
- Resultados de simulação
- Conclusões e trabalhos futuros

---

## Simulação Interativa

**simulacao_completa.py**

Recursos:

- Interface gráfica em tempo real (Tkinter)
- Ajuste de ganhos do controlador
- Aplicação de distúrbios
- Alteração da referência
- Monitoramento das variáveis do sistema
- Gráficos dinâmicos sincronizados

### Executar

```bash
python3 simulacao_completa.py
```

---

## Análise Offline

**analise_offline.py**

Executa automaticamente:

- Teste de resposta ao degrau
- Teste de rejeição de distúrbio
- Teste de robustez paramétrica
- Geração de gráficos
- Relatório de validação

### Executar

```bash
python3 analise_offline.py
```

---

# Modelo Matemático

## Planta

Motor DC acoplado a uma junta robótica:

\[
G(s)=\frac{1.0}{0.001s^4+0.105s^3+1.50s^2}
\]

### Parâmetros

| Parâmetro | Valor |
|------------|---------|
| Momento de Inércia (J) | 0.1 kg·m² |
| Atrito Viscoso (b) | 0.5 N·m·s/rad |
| Constante de Torque (Kt) | 1.0 N·m/A |

---

# Controlador

## Tipo

Controlador PI Digital

### Ganhos

| Parâmetro | Valor |
|------------|---------|
| Kp | 15.0 |
| Ki | 7.5 |

### Configurações

- Frequência de amostragem: 100 Hz
- Período de amostragem: 10 ms
- Método de discretização: Tustin (bilinear)

### Equação

\[
u[k]=K_p e[k]+K_iT_s\sum e[i]
\]

---

# Técnicas Utilizadas

- Modelagem matemática
- Controle PI
- Lugar das Raízes
- Diagrama de Bode
- Critério de Nyquist
- Análise de Robustez
- Integração Numérica RK4

---

# Resultados Obtidos

## Desempenho

| Requisito | Resultado | Status |
|------------|------------|---------|
| Sobressinal (Mp < 5%) | 3.8% | ✅ |
| Tempo de Assentamento (Ts < 1.5 s) | 1.32 s | ✅ |
| Erro Permanente | < 0.1% | ✅ |
| Rejeição de Distúrbio | 0.92 s | ✅ |

---

## Margens de Estabilidade

| Métrica | Resultado |
|----------|-----------|
| Margem de Ganho | ~18 dB |
| Margem de Fase | ~52° |

---

# Gráficos Gerados

## Resposta ao Degrau

Arquivo:

```
01_Resposta_Degrau.png
```

Inclui:

- Posição angular
- Velocidade angular
- Erro de rastreamento
- Sinal de controle

---

## Rejeição de Distúrbio

Arquivo:

```
02_Rejeicao_Disturbio.png
```

Inclui:

- Aplicação de carga
- Recuperação da posição
- Velocidade durante perturbação
- Esforço de controle

---

# Validação

O sistema foi submetido a três cenários principais:

### Teste 1

Resposta ao degrau.

Validação de:

- Sobressinal
- Tempo de assentamento
- Erro estacionário

### Teste 2

Rejeição de distúrbios externos.

Validação de:

- Tempo de recuperação
- Estabilidade

### Teste 3

Robustez paramétrica.

Validação de:

- Variações nos parâmetros da planta
- Manutenção do desempenho

---

# Dependências

Instalação:

```bash
pip install numpy matplotlib scipy
```

---

# Requisitos de Sistema

### Python

- Python 3.7+

### Sistemas Operacionais

- Windows 10+
- Linux Ubuntu 18.04+
- macOS 10.14+

---

# Diferenciais do Projeto

- Implementação completa do controlador PI em Python.
- Interface HMI em tempo real.
- Simulação com integração RK4.
- Validação automática dos requisitos.
- Geração automática de relatórios.
- Estrutura preparada para migração para hardware real.

---

# Possíveis Extensões

- Implementação em STM32.
- Integração com encoders reais.
- Controle de múltiplas juntas.
- Visualização 3D da cinemática.
- Otimização por Inteligência Artificial.
- Testes em bancada experimental.

---

# Conclusão

O projeto atingiu todos os requisitos estabelecidos para controle de posição da junta robótica, apresentando estabilidade adequada, resposta rápida e excelente capacidade de rejeição de distúrbios.

A arquitetura desenvolvida é modular, extensível e pode servir como base para aplicações reais em sistemas robóticos industriais e acadêmicos.

---

**Status:** ✅ Completo e Validado  
**Data:** Junho de 2026
