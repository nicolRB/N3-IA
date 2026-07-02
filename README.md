# Classificador de Imagens com CNN do Zero — CIFAR-10

Demonstração prática do trabalho de **Visão Computacional** (Seminários — Aplicações Inteligentes no Dia a Dia, N3).

## O que é

Uma Rede Neural Convolucional (CNN) construída **do zero** (sem transfer learning), treinada para classificar imagens do dataset **CIFAR-10** em 10 categorias: avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio e caminhão.

## Como executar

### Opção 1 — Google Colab (recomendado)

1. Acesse [colab.research.google.com](https://colab.research.google.com/). ([Notebook Executado](https://colab.research.google.com/drive/1hV42dtdL5kn28zdwagQPFA-v8J7-6WgR#scrollTo=d56e9032))
2. Vá em **Arquivo → Fazer upload de notebook** e envie o arquivo `classificador-cnn-cifar10.ipynb`.
3. Ative a GPU em **Ambiente de execução → Alterar tipo de ambiente de execução → GPU**.
4. Rode todas as células em **Ambiente de execução → Executar tudo**.

Todas as bibliotecas usadas (`tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) já vêm pré-instaladas no Colab.

### Opção 2 — Ambiente local (Jupyter)

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn notebook
jupyter notebook classificador-cnn-cifar10.ipynb
```

> O treinamento é bem mais rápido com GPU. Em CPU, as 15 épocas podem levar vários minutos.

## Estrutura do notebook

1. Importação das bibliotecas
2. Carregamento e exploração do dataset (CIFAR-10)
3. Pré-processamento (normalização)
4. Definição da arquitetura da CNN
5. Compilação e treinamento
6. Curvas de treinamento (acurácia e perda)
7. Avaliação no conjunto de teste
8. Matriz de confusão
9. Demonstração de predição em novas imagens
10. Salvamento do modelo treinado
11. Conclusão e possíveis melhorias

## Arquitetura da rede

```
Entrada (32x32x3)
   ↓
Conv2D (32 filtros, 3x3) + ReLU
MaxPooling2D (2x2)
   ↓
Conv2D (64 filtros, 3x3) + ReLU
MaxPooling2D (2x2)
   ↓
Conv2D (64 filtros, 3x3) + ReLU
   ↓
Flatten
Dense (64) + ReLU
Dropout (0.5)
Dense (10) + Softmax
```

## Resultado esperado

Acurácia entre **70% e 80%** no conjunto de teste após ~15 épocas — resultado didático que evidencia tanto o potencial quanto as limitações de uma CNN simples, sem data augmentation ou arquiteturas pré-treinadas.

## Autores

Trabalho da disciplina de Inteligência Artificial — Prof. Claudinei Dias (Ney).
