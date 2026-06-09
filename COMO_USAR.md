# Como Usar e Iniciar a Aplicação

Este documento explica de forma simples como a aplicação funciona e como iniciá-la no seu navegador.

## 🛠️ Como funciona?

A aplicação é uma ferramenta de **Diagnóstico de Fissuras Estruturais**. Ela utiliza técnicas de visão computacional e inteligência artificial para analisar fotos de fissuras e determinar a gravidade do problema.

O fluxo de uso é bem simples:
1. **Envio da Imagem:** Você faz o upload de uma foto da fissura (do seu computador) ou tira uma foto na hora usando a webcam.
2. **Calibração de Escala:** A ferramenta pode usar um marcador "ArUco" (um código quadrado impresso na cena) para descobrir o tamanho real das coisas na foto, ou usar uma medida manual configurada por você.
3. **Análise e Diagnóstico:** O sistema processa a imagem para isolar a fissura (ignorando rejuntes, manchas, etc.), mede a sua largura em milímetros e avalia o risco estrutural.
4. **Resultado Final:** A tela apresenta o diagnóstico completo, classificando a situação como **CRÍTICA**, **SEMI-CRÍTICA**, **NÃO-CRÍTICA** ou **DESCARTADA**, acompanhado de todas as medidas detectadas.

Tudo isso funciona através de uma interface interativa.

---

## 💾 Download dos Datasets (Para a Banca)

O modelo de Inteligência Artificial já está treinado e os seus pesos encontram-se na pasta `src/model_weights/`. **Não é necessário treinar o modelo novamente para rodar a aplicação.**

No entanto, caso os avaliadores queiram inspecionar as **imagens rotuladas** utilizadas durante o treinamento, o dataset não foi incluído diretamente neste repositório por exceder os limites de tamanho do GitHub.

Para acessar o dataset rotulado:
1. Faça o download do arquivo ZIP no link a seguir: `[COLE O LINK DO SEU GOOGLE DRIVE AQUI]`
2. Extraia o conteúdo baixado dentro da pasta raiz do projeto, certificando-se de que a pasta `dataset_rotulado` fique no local correto.

---

## 🚀 Como iniciar a aplicação

Para abrir a aplicação e usá-la, o processo é executado pelo terminal. Siga estes passos:

### Passo 1: Abrir o terminal na pasta correta
Você precisa abrir o terminal (Pode ser o Prompt de Comando, PowerShell ou o terminal da sua IDE) na pasta onde o projeto está salvo. 

Se estiver abrindo um terminal novo, navegue até a pasta do projeto com o comando:
```cmd
cd C:\caminho\ficticio\para\a\pasta\do\projeto
```

### Passo 2: Executar o comando de inicialização
Com o terminal aberto na pasta do projeto (onde está o arquivo `app.py`), digite o seguinte comando e aperte **Enter**:

```cmd
streamlit run app.py
```

### O que vai acontecer depois disso?
- O sistema iniciará um servidor local na sua máquina.
- **Automaticamente**, o seu navegador padrão (Google Chrome, Edge, Firefox, etc.) será aberto em uma nova aba carregando a aplicação.
- Caso o navegador não abra sozinho por algum motivo, você verá no terminal um link (geralmente `http://localhost:8501`). Basta copiar esse link e colar no seu navegador.

### Para parar a aplicação
Quando quiser parar de usar e encerrar o servidor, basta ir no terminal onde o comando está rodando e apertar `Ctrl + C`.
