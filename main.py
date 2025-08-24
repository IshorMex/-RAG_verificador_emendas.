import pandas as pd
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer
import torch

# INCIALIZANDO...
llm = pipeline("text2text-generation", model="unicamp-dl/ptt5-base-portuguese-vocab", tokenizer=T5Tokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab"), max_new_tokens=2048)
model = SentenceTransformer("all-mpnet-base-v2") 
# MODELOS ON!

def consultar(caso: str, maior_k: int):
    ramo_ementa = detectar_ramo(caso) #detectamos o ramo da emenda fornecida pelo usuario.
    
    if ramo_ementa not in indices_ramos:#caso nenhum dos ramos seja relacionado
        print("\nNenhum índice relevante encontrado para este ramo jurídico. :( )") 
        return

    #Damo inicio ao nosso metodo de "Filtragem"

    consultar_embedding = model.encode([caso], convert_to_numpy=True)
    faiss.normalize_L2(consultar_embedding)
    
    #buscando indicie e corpus especificos referentes ao ramo da emenda!
    index_especifico = indices_ramos[ramo_ementa]
    corpus_especifico = corpus_ramo[ramo_ementa]
    
    #busca por similaridade usando FAISS
    pontos, indices = index_especifico.search(consultar_embedding, maior_k)
    
    #definimos os limtes minimos de valor de similaridade para cada tipo especifico de ramo!
    limite = limitacao_dinamica(ramo_ementa)
    ementas_referentes = []
    
    #Filtragem de resultados!
    for pontuacao, idx in zip(pontos[0], indices[0]):
        if pontuacao >= limite:
            s = corpus_especifico[idx]
            ementas_referentes.append({"score": pontuacao, "text": f"Súmula {s['numero']} - {s['titulo']}: {s['enunciado']}"})
            #caso queira ver as sulmulas que ele encontrou com uma pontuacao agradavel retire o """" abaixo:
            print(f"[Pontuação: {pontuacao}]")
            print(f"Sumula Encontrada: Súmula {s['numero']}")
            print(f"Título: {s['titulo']}")
            print(f"Enunciado: {s['enunciado']}")
            print('-'*25)#linha para melhorar visualização
            

    #Selecionamos as 5 mais relevantes entao as entregamos a ML!
    if ementas_referentes:
        top5_ementa = sorted(ementas_referentes, key=lambda x: x['score'], reverse=True)[:5]
        top5 = [item['text'] for item in top5_ementa]
        print("Resposta do modelo....")
        print(gerar_resposta(caso, top5))
    else:
        print("\nNenhuma súmula relevante encontrada! :( )")

def detectar_ramo(ementa_analise: str) -> str:
    #Eh a mesma função utilizada no Scraping para definir o tipo de ramo de uma emenda(nesse caso a do usuario)
    ementa_analise = ementa_analise.lower()
    ramos = {"DIREITO PENAL": r"(penal|crime|prisão|furto|roubo|homicídio|latrocínio|tráfico|drogas|estelionato|pena|recurso especial|habeas corpus|condenação|absolvição|dosimetria|roubo|legítima defesa|roubo majorado|crime contra o patrimônio|crime contra a vida|violência doméstica)","DIREITO CIVIL": r"(civil|obrigações|contratos|responsabilidade civil|família|sucessões|dano moral|locação|imóveis|propriedade|usucapião|herança|inventário|divórcio|guarda|alimentos|consumidor|compra e venda|acidente de trânsito|reparação de danos)","DIREITO PREVIDENCIÁRIO": r"(previdênci|aposentadoria|inss|benefício|auxílio-doença|pensão por morte|aposentadoria por invalidez|salário-maternidade|contribuição previdenciária|tempo de serviço|carência|regime geral)","DIREITO ADMINISTRATIVO": r"(administrativo|servidor público|licitação|concurso|improbidade|desapropriação|poder público|ato administrativo|processo administrativo|multa administrativa|serviço público|licitações e contratos)","DIREITO TRIBUTÁRIO": r"(tributário|imposto|taxa|contribuição|icms|irpj|cide|pis|cofins|dívida ativa|execução fiscal|lançamento tributário|isenção fiscal|bitributação)"}
    for ramo, padroes in ramos.items():
        if re.search(padroes, ementa_analise):
            return ramo
    return "GERAL"

def limitacao_dinamica(ramo):
    #entregamos limites diferentes em casos diferentes! Com mais pesquisas/testes eh possivel refinar ainda mais esses valores!
    thresholds = {"DIREITO PENAL": 0.70, "DIREITO CIVIL": 0.65, "DIREITO PREVIDENCIÁRIO": 0.65,"DIREITO ADMINISTRATIVO": 0.60, "DIREITO TRIBUTÁRIO": 0.60, "GERAL": 0.60 }

    #como ñ houve ainda um grande refinamento estamos retornado um valor "medio" entre os que ja decidimos para outros casos.
    return thresholds.get(ramo, 0.60)

def gerar_resposta(caso, sumulas_relevantes): #mágica da ML
    sumulas_txt = "\n\n".join(sumulas_relevantes)

    #ROTEIRO, seguindo o que nos foi pedido acabei obtando por esse prompt engineering.Temos uma definição de persona, dados de entrada, instruções de formato de resposta, e adicionamos alguns 'few-shot' (exemplos) junto de um marcador final.
    prompt = f"""Você é um especialista em direito do STJ e sua tarefa é verificar se a ementa de um caso se enquadra em uma das súmulas fornecidas.

    ## EMENTA:
    {caso}

    ## SÚMULAS FORNECIDAS:
    {sumulas_txt}

    Instruções:
    1. A lista de súmulas fornecida está ordenada da mais relevante para a menos relevante. Priorize a análise das primeiras súmulas da lista.
    2. Avalie a ementa e compare-a com as súmulas fornecidas.
    3. Responda estritamente no formato 'Sim: <Súmula> - <Justificativa>' se a ementa se enquadra em uma súmula e cite a súmula e a sua justificativa.
    4. Responda estritamente no formato 'Não: <Justificativa>' se a ementa não se enquadra em nenhuma súmula e justifique o porquê.
    5. Não adicione nenhum preâmbulo, saudação ou texto adicional. Sua resposta deve começar com 'Sim:' ou 'Não:'.

    Exemplo 1 (Sim):
    EMENTA: A decisão de prisão preventiva para assegurar a aplicação da lei penal deve ser justificada com base em fatos concretos do caso.
    SÚMULAS: Súmula 79 - A exigência de fiança deve ser justificada.
    RESPOSTA: Sim: Súmula 79 - A súmula trata da necessidade de justificar medidas cautelares, o que se alinha com a necessidade de fundamentação da prisão preventiva.

    Exemplo 2 (Não):
    EMENTA: A compra de um veículo com vício oculto enseja a responsabilidade civil do vendedor.
    SÚMULAS: Súmula 221 - São civilmente responsáveis pelo ressarcimento de dano, decorrente de publicação pela imprensa, tanto o autor do escrito quanto o proprietário do veículo de divulgação.
    RESPOSTA: Não: A ementa trata de vício em compra e venda de veículo, enquanto a súmula aborda a responsabilidade por danos causados por publicação na imprensa.

    ### RESPOSTA:"""
    
    #chamamos o modelo (unicamp-dl/ptt5-base-portuguese-vocab) um modelo de base T5, logo sua resposta infelzmente ñ é a mais limpa...Caso houvesse acesso para uso de bots melhores (NVIDIA LLM,SABIA-7B ou GPT-3.5) o resultado seria ainda melhor!!
    resposta_completa = llm(prompt)[0]["generated_text"]
    return resposta_completa

#---------------------------------------------
# extrair o .json para uso
data_coletado = pd.read_json("sumulas_embeddings.json")

# Utilizei de um metodo de agrupamento de súmulas por ramo jurídico
ramos_unicos = data_coletado['ramo'].unique()
indices_ramos = {}
corpus_ramo = {}

#Organizaremos nossos dados.json + Metodo FAISS utilizando corpus(principalmente ramo) + indices 
#ou seja, um processamento e organização dos dados tendo em mente seus Ramos Jurídicos
for ramo in ramos_unicos:
    
    sub_data_coletada = data_coletado[data_coletado['ramo'] == ramo] #cria-se uma tabela (como se fosse um sub-data temporario) com sumulas apenas do 'ramo' atual!.

    #PREPARACAO DO CORPUS E DOS VETORES!!!

    #salva o corpus para uso pelo LLM e Extrai os embeddings para o FAISS!!!
    corpus_ramo[ramo] = sub_data_coletada[["numero", "titulo", "enunciado"]].to_dict(orient="records")
    embeddings_ramo = np.array(sub_data_coletada["embedding"].tolist()).astype("float32")
    
    #Criando e adicionando índicies FAISS
    if embeddings_ramo.size > 0:
        # normaliza os vetores para o uso de busca por similaridade de cosseno.
        faiss.normalize_L2(embeddings_ramo)
        dimensao = embeddings_ramo.shape[1]

        #cria um novo indice FAIS para o ramu atual. 
        index_ramo = faiss.IndexFlatIP(dimensao)
        #add do ramo ao índice. 
        index_ramo.add(embeddings_ramo)


        indices_ramos[ramo] = index_ramo
        #Assim criamos um índice FAISS especializado para cada ramo!
    else:
        print(f"Aviso: Nenhum embedding encontrado para o ramo '{ramo}'.")

ementa = input("Envie uma ementa para ser analisada: ")
print()
K =  50
#Por mais bobo que pareca esse 'K' eh muito importante, eh recomendado utiliza-la entre 25 - 50 nessa ML! O mais convencional (5-15) acaba por n ser muito efetivo nesse prototipo D:
consultar(ementa, K)
