# Detecção Facial + Diário Emocional (OpenCV + MediaPipe)

## 🎯 Objetivo
Aplicação local que detecta rosto e estima emoção (**feliz / neutro / raiva / negativo**) em tempo real, exibindo overlay na tela e registrando um **diário emocional** em CSV.  
Foco: auxiliar usuários com **impulso em jogos online** a reconhecer estados emocionais e adotar pausas.

---

## 🛠 Tecnologias
- **OpenCV** → captura da webcam e exibição dos overlays.  
- **MediaPipe Face Mesh** → detecção de landmarks faciais (boca, sobrancelhas, nariz, etc.).  
- **NumPy** → cálculos geométricos simples.  

---

## ▶️ Como executar

### 1. Criar ambiente virtual
```bash
python -m venv .venv
```

### 2. Ativar o ambiente
- **Windows (PowerShell)**  
  ```powershell
  .venv\Scripts\activate
  ```
- **Linux/macOS**  
  ```bash
  source .venv/bin/activate
  ```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Executar aplicação
```bash
python app.py
```

👉 Pressione **ESC ou Q** para encerrar.  
👉 Ao final, será salvo um arquivo **`emotional_log.csv`** com o resumo da sessão.

---

## 🎚 Parâmetros (sliders)

Durante a execução, a janela exibe sliders para ajustar a sensibilidade do modelo:

- **TH_SMILE, TH_FROWN** → controlam detecção de sorriso e “cantos caídos” da boca.  
- **MOUTH_OPEN_LOW, MOUTH_OPEN_HIGH** → limites de abertura da boca (MAR – Mouth Aspect Ratio).  
- **JANELA_SEG** → tamanho da janela de suavização temporal (frames usados para cálculo).  
- **NEG_RATIO, ALERTA_MS** → controlam quando sugerir pausa (proporção de emoções negativas e tempo mínimo).  
- **LOG_INT seg** → frequência de registro no diário (em segundos).  

---

## 🎥 Demonstração do impacto
- Ajuste **TH_SMILE** → sorrisos leves passam a ser classificados como “feliz”.  
- Aumente **NEG_RATIO** → menos banners de pausa são exibidos.  
- Diminua **ALERTA_MS** → alertas aparecem mais rápido.  
- Em caso de **raiva detectada** → aparece um **banner vermelho** na tela.  

video: https://youtu.be/gaItFyAYU20

---

## ⚠️ Limitações
- Heurísticas simples → não representam diagnóstico clínico.  
- Suscetível a condições de iluminação, qualidade da câmera e variações individuais.  
- Pode apresentar **viés** em diferentes perfis demográficos.  
- **Não substitui acompanhamento profissional** de saúde mental.  

---

## 🚀 Próximos passos
- Calibração por usuário (**baseline individual**).  
- Novas features → uso de sobrancelhas e olhos para melhorar heurísticas.  
- Interface gráfica para exportar e visualizar histórico.  
- Conteúdos de **micro-pausa** co-criados com profissionais de psicologia/saúde.  

---

## 📜 Nota ética (uso de dados faciais)
- Todo o processamento é **local**, nenhum dado é enviado para a nuvem.  
- O uso da câmera é **opt-in** e pode ser desligado a qualquer momento.  
- Os logs são **locais e opcionais**; o usuário pode apagar o CSV quando desejar.  
- O sistema **não rotula condições clínicas**. É uma ferramenta de apoio e conscientização, **não tratamento**.  
