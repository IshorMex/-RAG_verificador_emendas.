<img width="512" height="116" alt="image" src="https://github.com/user-attachments/assets/9a36285e-cae9-437c-b17d-bbb1f8b68a30" />

# ⚖️ Sumula Jurídica STJ: Ferramenta de Análise de Súmulas

Este projeto é uma ferramenta de análise de texto jurídico que utiliza técnicas de Processamento de Linguagem Natural (PLN) e busca por similaridade semântica para encontrar súmulas relevantes do Superior Tribunal de Justiça (STJ). O sistema utiliza um modelo de linguagem para gerar uma análise sobre o enquadramento de uma ementa fornecida pelo usuário.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-%23E87937.svg?style=for-the-badge&logo=HuggingFace&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-005C99?style=for-the-badge&logo=faiss&logoColor=white)
![Playwright](https://img.shields.io/badge/Playwright-2F80ED?style=for-the-badge&logo=playwright&logoColor=white)

---
✰ Recursos
-
⦁	**Coleta de dados Automática**: Usa Playwright e BeautifulSoup para fazer web scraping das súmulas do STJ.
⦁	**Busca por Similaridade**: Utilizando do FAISS para buscas rápidas e eficientes de súmulas semanticamente semelhantes.
⦁	**Filtragem por Ramo Jurídico**: Classifica as súmulas e a ementa do usuário em ramos como Direito Penal, Direito Civil, entre outros, para otimizar a busca.
⦁	**Análise de Enquadramento**: Utiliza o modelo de linguagem "unicamp-dl/ptt5-base-portuguese-vocab" para gerar uma resposta concisa, indicando se a ementa se enquadra em uma súmula e justificando a conclusão.

---

⚙︎ Instalação e Configuração
-

⦁	**pré-requisitos** - Certifique-se de ter o Python 3.8 ou superior instalado:
	- Instale as dependências:
 
    pip install pandas numpy faiss-cpu sentence-transformers transformers torch playwright beautifulsoup4 rich

  -	Instale os navegadores necessários para o Playwright:
```
playwright install
```
---
▶︎ Como Executar o Projeto
-
O projeto é composto por duas etapas principais: a coleta e o processamento inicial dos dados (scrapping_embedding.py) e a execução da ferramenta de consulta (main.py).Caso baixe o .json já incluído no Files não é necessário utilizar do método de processamento de dados, entretanto, você não tera acesso a atualizações após o dia de upload(24/08/2025)

⦁	**Passo 1: Coletar e Processar as Súmulas**<br>
	Execute o script de web scraping para coletar as súmulas do STJ e gerar o arquivo de dados com os embeddings.
  ```
  python ./scrapping_embedding.py
  ```

  Este script criará o arquivo sumulas_embeddings.json no mesmo diretório. Este arquivo é essencial para a próxima etapa.


⦁	**Passo 2: Analisar uma Ementa**<br>
	Execute o script principal. Ele carregará o arquivo 'sumulas_embeddings.json' e pedirá uma ementa para análise.
  ```
  python ./main.py
  ```

  Quando solicitado, insira uma ementa, como no exemplo abaixo:
  <br>
```	"Envie uma ementa para ser analisada: A prescrição em matéria de direito penal não se aplica quando o crime for continuado."```

---
☼ Como Funciona
-
A ementa fornecida é classificada em um ramo jurídico (detectar_ramo). Então, um vetor (embedding) é gerado para a ementa usando o modelo de linguagem. O FAISS em seguida, busca no índice específico daquele ramo jurídico as súmulas mais similares à ementa. As súmulas encontradas com um nível de similaridade acima de um limite pré-definido são filtradas. Assim, as 5 súmulas mais relevantes são passadas para o modelo de linguagem PTT5. Por fim o modelo gera uma resposta formatada, indicando se a ementa "se enquadra" em uma das súmulas e fornecendo uma justificativa.

---
༄ Contribuições
-
Contribuições são bem-vindas! Se você tiver sugestões para melhorar a detecção de ramos, os limites de similaridade (limitacao_dinamica) ou a prompt engineering do LLM, sinta-se à vontade para abrir uma issue ou enviar um pull request.

---
[<img loading="lazy" src="https://avatars.githubusercontent.com/u/102492011?v=4" width=115><br><sub>Joao Pedro Hupeps Arenales </sub>](https://github.com/IshorMex)
