import pandas as pd
import numpy as np
import faiss
import re
from playwright.sync_api import sync_playwright, Playwright
from sentence_transformers import SentenceTransformer
from rich import print
from bs4 import BeautifulSoup

#INICIALIZACAO...
model = SentenceTransformer("all-mpnet-base-v2") # on, agora podemos usa-lo para criar embeddings a partir do texto!
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}  # isso ajuda a simular uma requisição de navegador mais realista para evitar bloqueios.

def detectar_ramo(analise):

    #analisamos um texto e retornamos o ramo juridico mais provavel com base em palavras-chaves que tenham relação com o ramo. Caso nenhum ramo for detectado, retornamos "GERAL"!

    analise = analise.lower()
    ramos = {"DIREITO PENAL": r"(penal|crime|prisão|furto|roubo|homicídio|latrocínio|tráfico|drogas|estelionato|pena|recurso especial|habeas corpus|condenação|absolvição|dosimetria|roubo|legítima defesa|roubo majorado|crime contra o patrimônio|crime contra a vida|violência doméstica)","DIREITO CIVIL": r"(civil|obrigações|contratos|responsabilidade civil|família|sucessões|dano moral|locação|imóveis|propriedade|usucapião|herança|inventário|divórcio|guarda|alimentos|consumidor|compra e venda|acidente de trânsito|reparação de danos)","DIREITO PREVIDENCIÁRIO": r"(previdênci|aposentadoria|inss|benefício|auxílio-doença|pensão por morte|aposentadoria por invalidez|salário-maternidade|contribuição previdenciária|tempo de serviço|carência|regime geral)","DIREITO ADMINISTRATIVO": r"(administrativo|servidor público|licitação|concurso|improbidade|desapropriação|poder público|ato administrativo|processo administrativo|multa administrativa|serviço público|licitações e contratos)","DIREITO TRIBUTÁRIO": r"(tributário|imposto|taxa|contribuição|icms|irpj|cide|pis|cofins|dívida ativa|execução fiscal|lançamento tributário|isenção fiscal|bitributação)"}

    for ramo, padroes in ramos.items():
        if re.search(padroes, analise):
            return ramo
    return "GERAL"

def run (playwright: Playwright):

    #nossa automização de coleta de súmulas do STJ por meio do PLAYWRIGHT  e do BEAUTIFULSOUP. No fim, retornamos uma lista de dicionarios com os dados dos súmulas!

    ulr_inicio = "https://scon.stj.jus.br/SCON/sumstj/toc.jsp?documentosSelecionadosParaPDF=&numDocsPagina=100&tipo_visualizacao=&data=&p=false&tipo=sumula&b=SUMU&i=1&l=10&ordenacao=%40NUM&operador=E" 

    #simulamos um navegador Chrome, com o playwright!
    chrome = playwright.chromium
    browser = chrome.launch(headless = True)
    pagina = browser.new_page()
    pagina.goto(ulr_inicio)

    corpus = []

    #inicia o loop para coleta de dados em todas as páginas do site.
    while True:
        #usamos do BeautifulSoup4 para pegar o conteduo HTML do site
        pagina_cont = pagina.content()
        soup = BeautifulSoup(pagina_cont,"html.parser")
        #sopa ^^

        #procura por todos os 'blocos' de sumulas na pagina atual
        bloco_sumulas = soup.find_all(class_="gridSumula")
        print("Blocos encontrados:", len(bloco_sumulas))
        for bloco in bloco_sumulas:
            #O site utiliza o texto "CANCELADA" para sumulas que foram revogadas (ou seja, ñ tem mais efeito legal).Ignoraremos essas.
            if "CANCELADA" not in bloco.get_text():

                #extraimos o titulo, enunciado e numero da sumulas
                titulo = bloco.find(class_="ramoSumula")
                enunciado = bloco.find(class_="clsVerbete")
                numero_sumu = int(bloco.find("span",class_="numeroSumula").get_text().strip())


                #aqui ha um plano B para as sumas apartir da 655, onde o padrão do HTML muda.
                if not enunciado: 
                    enunciado = titulo.next_sibling


                
                #por precaucao verificamos se a coleta foi um sucesso.
                if titulo and enunciado:
                    # para acompanhar o progresso da coleta remova o '#' da linha abaixo:
                    
                    # print(f"Coletando súmula {numero_sumu} - {titulo.text.strip()}")

                    #por fim classificamos nossa sulma coletda em algum ramo!
                    ramo = detectar_ramo(enunciado.get_text().strip())

                    #organizamos as informações em um dicionario para cada sumula em formato de ficha.
                    sumula = {
                        "numero": numero_sumu,
                        "titulo": titulo.text.strip(),
                        "enunciado": enunciado.get_text().strip(),
                        'ramo': ramo
                    }
                    #cada ficha eh adicionada no corpus.
                    corpus.append(sumula) 
                
                #condicao de paradada: chegar na ultima sumula 676.
                if numero_sumu == 676:
                    print(f"Limite de 676 súmulas atingido. Parando a coleta.")
                    return corpus
                    
        #seguimos para coleta na proxima pagina
        proximo_botao = pagina.locator('a.iconeProximaPagina[data-bs-original-title="Próxima página"]').first
        
        #'precaucao' caso a linha 77 ñ funcione como o esperado.        
        if proximo_botao.is_visible():
            #Playwright clica no botão para a próxima página! reiniciando o loop de coleta
            proximo_botao.click()
            pagina.wait_for_load_state("networkidle")
        else:    
            break
    return corpus

#-------------------------------------------------------------------------------------------------------

#PONTO DE INICIO DO PROGRAMA!
# #utilizamos da biblioteca "playwright" para controlar um navegador e iniciar a coleta!
with sync_playwright() as playwright:

    #corpus sera preechido pela coleta do playwright na função run
    corpus = run(playwright)

    print("total de sumulas coletadas:", len(corpus))
    
    #por enquanto salvaremos nosso progresso até então :D para usarmos um .json mais tarde na main.

    #precaucao para caso o corpus esteja vazio (ou seja houve um ERRO no programa).
    if len(corpus) > 0:
        #embedding para cada 'enunciaado' no corpus (metodo usado = SentenceTransformer)
        embeddings = model.encode([c["enunciado"] for c in corpus], convert_to_numpy=True)

        #acrescentamos os embeddings a nosso corpus.
        for i, emb in enumerate(embeddings):
            corpus[i]["embedding"] = emb.tolist()
        
        #converte o corpus para um DataFrame do pandas, e então os salvamos em um arquivo .JSON
        pd.DataFrame(corpus).to_json("sumulas_embeddings.json", orient="records", indent=2)
    else:
        print("Nenhuma súmula encontrada!Verifique-se o codigo esta correto")