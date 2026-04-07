Este é o manual definitivo de operação do seu **Sistema de Projeção Interativa PRO**. 

Ele foi desenhado para ser um motor de jogo em tempo real que transforma sua mesa em uma arena digital com rastreamento físico de latência zero.

---

# 📖 MANUAL DE OPERAÇÃO: PROJEÇÃO INTERATIVA

## 1. INICIALIZAÇÃO E COMANDOS RÁPIDOS
Para rodar, no terminal use: `python painel.py`

### Atalhos de Teclado (Foco na Janela da Câmera):
*   **Q ou ESC:** Morte Súbita (Fecha todas as janelas e mata processos de IA no fundo).
*   **ESPAÇO:** Registra Ponto de Calibração (Use 4 cliques conforme orientação na mesa).
*   **CLIQUE + ARRASTE (Mouse):** Cria uma caixa de rastreio em volta de qualquer objeto físico na mesa.

---

## 2. PAINEL DE CONTROLE (CORE DASHBOARD)

### 🤖 VISÃO IA: ON/OFF
*   **OFF (Desejado para o Monitor):** Esconde os efeitos, rastros e insetos do seu monitor de PC para você ter uma visão limpa da câmera. Os efeitos **continuam aparecendo na mesa do projetor**.
*   **ON (Modo Debug):** Espelha tudo o que o projetor está desenhando também na tela do seu PC. Útil para verificar se o sistema está "enxergando" o objeto corretamente.

---

## 3. CALIBRAÇÃO (ALINHAMENTO MÁXIMO)
Para que o gráfico saia **exatamente** em cima do objeto na mesa, siga estes passos:
1. Clique em **CALIBRAR SISTEMA** no painel de botões.
2. Quatro cruzes aparecerão na projeção (mesa).
3. **Pelo PC (Mouse):** Clique no centro exato de cada cruz conforme elas aparecem (Superior Esquerda -> Superior Direita -> Inferior Direita -> Inferior Esquerda).
4. O sistema salvará o arquivo `calibration.npy`. Da próxima vez que abrir, ele já sabe onde a mesa está!

---

## 4. SISTEMA DE MEMÓRIA E IDENTIDADES
1. Selecione um objeto com o mouse.
2. Clique em **MEMÓRIA / CONFIGS**.
3. No campo de nome, dê um apelido (Ex: "LATINHA").
4. Clique em **Salvar Identidade Inteligente**.
*   **Efeito:** O sistema agora conhece a textura, cor e forma desse item. Se você tirar o objeto da mesa e colocar de novo, ele pode ser re-identificado automaticamente via Visão Computacional.

---

## 5. REFINAMENTO DE EFEITOS
No menu **EFEITOS (FX)**, você tem 12 modos diferentes, incluindo:
*   **FOGO CÓSMICO:** Chamas persistentes com rastro químico.
*   **RAIO PLASMÁTICO:** Descargas elétricas caóticas entre os pontos.
*   **TEIA PLEXUS:** Conecta objetos físicos por fios de rede digital.
*   **CONTROLE DE FADE:** O Slider de "Velocidade de Desvanecimento" define se o rastro na mesa some rápido ou se fica pintado lá por muito tempo (0 = Permanente).
*   **ATAQUE DE INSETOS (JOGO):** Ao ativar, aranhas virtuais surgem nas bordas e perseguem seus objetos físicos selecionados.
---

## 💡 DICAS DE OURO
*   **Iluminação:** O sistema funciona melhor com luz controlada. 
