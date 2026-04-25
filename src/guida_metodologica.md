# Guida Metodologica Completa
## Come funziona la pipeline di analisi dei margini sui carburanti italiani
### Spiegazione passo per passo per chi non ha mai letto nulla di statistica

---

> **Come leggere questo documento.** Ogni sezione spiega prima *cosa* fa un determinato
> passaggio, poi *perché* è stato scelto, poi *come si interpreta* il risultato.
> I numeri riportati vengono direttamente dall'output reale della pipeline.
> Le analogie non sono ornamentali: servono a rendere intuitivo qualcosa di astratto.

---

## Il problema che vogliamo risolvere

Quando scoppia una crisi energetica — guerra in Ucraina, conflitto Iran-Israele, chiusura
dello Stretto di Hormuz — i prezzi della benzina e del diesel alla pompa salgono. È normale:
il costo delle materie prime aumenta e i distributori lo trasferiscono ai consumatori.

La domanda che questa analisi si pone è più precisa: **i distributori italiani hanno
trasferito ai consumatori solo il maggior costo, oppure ne hanno approfittato per
allargare il proprio margine di guadagno rispetto a periodi normali?**

Per rispondere, non guardiamo il prezzo alla pompa, ma la *differenza* tra il prezzo alla
pompa (senza tasse) e il costo del carburante sul mercato all'ingrosso europeo. Questa
differenza si chiama **crack spread** ed è la proxy del margine lordo del distributore.

Se il crack spread durante la crisi è simile a quello del 2019 (anno "normale" pre-COVID,
con mercato maturo e Brent stabile), i distributori hanno fatto pass-through corretto.
Se è significativamente più alto, c'è un'anomalia che merita spiegazione.

---

## La struttura della pipeline: cinque passi in sequenza

La pipeline è composta da cinque script Python che si eseguono in ordine. Ciascuno
risponde a una domanda precisa e passa i risultati al successivo.

```
01_data_pipeline.py    → Raccoglie e pulisce i dati
02_changepoint.py      → Quando si è rotta la dinamica dei prezzi?
03_margin_hypothesis.py → Il margine è anomalo rispetto al 2019?
04_auxiliary_evidence.py → L'anomalia è specifica all'Italia?
05_global_corrections.py → Quanti risultati reggono dopo correzione statistica?
```

---

## PASSO 1 — Raccolta dati (`01_data_pipeline.py`)

### Cosa fa

Raccoglie da fonti pubbliche tre serie di dati settimanali:

- **Prezzo del Brent in euro/barile** — scaricato da Yahoo Finance (ticker BZ=F),
  convertito da USD a EUR usando il tasso di cambio della stessa settimana.
- **Prezzi alla pompa italiani senza tasse** — dall'EU Weekly Oil Bulletin, il bollettino
  ufficiale europeo che ogni lunedì pubblica i prezzi al consumo netti di accise e IVA
  per tutti i paesi EU.
- **Futures wholesale europei** — Eurobob (benzina) e Gas Oil ICE (diesel), che
  rappresentano il costo all'ingrosso che i distributori effettivamente pagano sul
  mercato spot di Amsterdam-Rotterdam-Anversa.

Il risultato è un dataset unificato di 381 settimane (gennaio 2019 – aprile 2026).

### Perché queste fonti

**Perché usare i prezzi alla pompa senza tasse?** Le tasse in Italia sono fisse
(accisa + IVA). Se confrontassimo i prezzi lordi, includeremmo una componente che
non dipende affatto dal comportamento del distributore. Togliere le tasse isola
la parte che riflette costo wholesale + margine.

**Perché usare i futures Eurobob/Gas Oil e non il Brent?** Il Brent è il prezzo del
greggio, non della benzina raffinata. Tra il greggio e la pompa ci sono costi di
raffinazione e distribuzione. I futures Eurobob e Gas Oil ICE sono invece i prezzi
effettivi a cui i distributori italiani comprano il prodotto già raffinato sul mercato
europeo. Usarli come proxy del costo wholesale è molto più accurato.

**Perché il 2019 come baseline?** Il 2019 è l'anno pre-crisi per eccellenza: niente
COVID, niente guerre energetiche, Brent stabile tra 60 e 70 dollari al barile,
mercato dei carburanti maturo e competitivo. È il nostro "termometro della normalità".

### Il crack spread: il margine lordo che analizziamo

Il crack spread si calcola così:

```
crack spread = prezzo pompa (netto tasse) − prezzo wholesale europeo
```

Dalla pipeline: la media del crack spread benzina nel 2019 è **0.168 EUR/litro**,
con deviazione standard di 0.019 EUR/litro. Quella del diesel è **0.149 EUR/litro**,
deviazione standard 0.018 EUR/litro.

Questi numeri diventano il nostro punto di riferimento assoluto.

---

## PASSO 2 — Changepoint bayesiano (`02_changepoint.py`)

### Cosa fa

Prima di testare se il *margine* è anomalo, vogliamo sapere quando esattamente i
*prezzi* hanno cominciato a muoversi rispetto alla normalità pre-crisi. Questo ci
serve per definire correttamente le finestre di analisi.

Il modello individua un **changepoint** τ (tau): la data in cui la traiettoria dei
log-prezzi ha subito una rottura strutturale. Il modello è "bayesiano", il che
significa che non restituisce solo un numero puntuale ma un'intera distribuzione
di probabilità su quando potrebbe essere avvenuta la rottura.

### Analogia: il termostato che si regola

Immaginate un grafico del prezzo del Brent nel tempo come una linea che sale e scende
dolcemente. Il changepoint è il momento in cui quella linea ha cambiato
*direzione sistematica* — non una piccola oscillazione, ma un cambiamento di pendenza
permanente. Il modello divide la serie in due tratti (prima e dopo τ) e trova il punto
di divisione che spiega meglio i dati.

### Perché un modello bayesiano e non più semplice

Un approccio banale sarebbe: "prendo la data dell'evento geopolitico (es. 24 feb 2022
per l'Ucraina) e quella è la mia soglia." Il problema è che i mercati finanziari sono
*forward-looking*: anticipano gli eventi. Quindi il prezzo può aver già cominciato a
muoversi settimane prima dell'evento formale.

Il modello bayesiano lascia che siano i *dati* a dirci quando è avvenuta la rottura,
senza imporla a priori. Usa una distribuzione StudentT (invece della normale) per i
residui, perché i prezzi energetici hanno "code pesanti" — movimenti estremi sono
più frequenti di quanto preveda una distribuzione normale.

### Il lag D: anticipo o ritardo?

Il **lag D = τ − data_shock** misura la distanza tra il changepoint stimato e la data
dell'evento geopolitico. Se D è negativo, il mercato aveva già anticipato lo shock.

Risultati reali dalla pipeline:

| Evento | Serie | τ stimato | Lag D | Convergenza MCMC |
|---|---|---|---|---|
| Ucraina (24 feb 2022) | Benzina | 3 gen 2022 | −52 giorni | ✓ Rhat=1.001 |
| Ucraina | Diesel | 3 gen 2022 | −52 giorni | ✓ Rhat=1.004 |
| Iran-Israele (13 giu 2025) | Benzina | 28 apr 2025 | −46 giorni | ✓ Rhat=1.002 |
| Iran-Israele | Diesel | 5 mag 2025 | −39 giorni | ✓ Rhat=1.001 |
| Hormuz (28 feb 2026) | Benzina | 2 mar 2026 | +2 giorni | ✓ Rhat=1.002 |
| Hormuz | Diesel | 23 feb 2026 | −5 giorni | ✓ Rhat=1.001 |

**Interpretazione:** per Ucraina e Iran-Israele il mercato aveva anticipato lo shock
di 39–73 giorni. I prezzi wholesale stavano già salendo ben prima dell'evento formale,
coerentemente con l'evidenza che i mercati futures incorporano aspettative geopolitiche.
Per Hormuz (crisi più improvvisa) il changepoint coincide quasi esattamente con l'evento.

### Cosa sono Rhat e MCMC

Il modello bayesiano stima la distribuzione del changepoint tramite simulazione
stocastica (**MCMC — Markov Chain Monte Carlo**): fa girare quattro "catene" di
campioni che esplorano lo spazio delle soluzioni possibili. **Rhat** (R-hat) misura
se le quattro catene hanno convergito alla stessa risposta: valori vicini a 1.00 indicano
convergenza, valori > 1.05 indicano problemi. Il Brent per Ucraina ha Rhat=1.162,
segnalato come "convergenza dubbia" — il risultato per quella serie va trattato con
maggiore cautela.

### I diagnostici OLS che motivano i test successivi

Come effetto collaterale del modello di regressione, il passo 2 produce tre diagnostici
sulla qualità statistica dei residui, che guidano la scelta dei test nel passo 3:

**Durbin-Watson (DW)** — misura l'autocorrelazione. Un valore normale è intorno a 2.
Valori molto bassi (es. DW = 0.29–0.42 osservati in quasi tutte le serie) indicano
autocorrelazione positiva forte: ciò che succede questa settimana è molto simile a
ciò che succedeva la settimana scorsa. Questo è un problema per i test classici.

**Shapiro-Wilk (SW)** — testa se i residui seguono una distribuzione normale. Se il
p-value SW è molto piccolo (< 0.05), i residui *non* sono normali. Osservato per
Ucraina benzina (p≈0) e diesel (p=0.0002), Hormuz benzina (p=0.0003).

**Breusch-Pagan (BP)** — testa se la varianza dei residui è costante nel tempo. Se non
lo è (eteroschedasticità), gli errori standard dei test sono distorrti. Osservato per
Iran diesel (p=0.021) e Hormuz benzina (p=0.0098).

---

## PASSO 3 — Test sull'anomalia del margine (`03_margin_hypothesis.py`)

Questo è il cuore dell'analisi. Risponde alla domanda: **il margine lordo nel periodo
post-shock è statisticamente diverso da quello del 2019?**

### L'ipotesi nulla e l'ipotesi alternativa

In statistica si parte sempre da una **ipotesi nulla (H₀)** — la situazione "niente di
speciale sta succedendo" — e si testa se i dati la contraddicono abbastanza da rifiutarla.

**H₀ (ipotesi nulla):** Il margine lordo medio nel periodo post-shock è uguale alla
media del 2019. In formule: μ_post = μ_2019.

**H₁ (ipotesi alternativa):** Il margine lordo medio nel periodo post-shock è
superiore alla media del 2019. In formule: μ_post > μ_2019.

Il test è **unilaterale superiore** (one-sided upper): ci interessa solo se il margine
è salito, non se è sceso. Un margine che scende non indica comportamento anomalo
a danno dei consumatori.

### La soglia di anomalia: ±2σ del 2019

Una variazione del margine può essere casuale (ogni settimana il margine oscilla
naturalmente) oppure sistematica (qualcosa di strutturale è cambiato). Per distinguerle
usiamo la **deviazione standard (σ) del 2019**: misura quanto oscillava normalmente
il crack spread in un anno tranquillo.

La soglia è posta a **2σ** sopra la media del 2019. Perché 2σ? In una distribuzione
normale, circa il 95% dei valori cade entro ±2σ dalla media. Se il margine post-shock
supera μ_2019 + 2σ, vuol dire che quel livello sarebbe stato eccezionale anche in un
anno normale — quindi non è spiegabile dalla variabilità ordinaria.

Valori dalla pipeline:
- Benzina 2019: μ = 0.168 EUR/L, 2σ = 0.038 EUR/L → soglia anomalia = 0.206 EUR/L
- Diesel 2019: μ = 0.149 EUR/L, 2σ = 0.037 EUR/L → soglia anomalia = 0.186 EUR/L

### Perché non basta un solo test

I diagnostici del passo 2 hanno rivelato tre problemi nelle serie di dati:
autocorrelazione forte, non-normalità in alcune serie, eteroschedasticità in altre.
Ogni problema viola le assunzioni di un test diverso. Usare un solo test rischierebbe
di ottenere risultati distorti.

La soluzione è usare una **batteria di test complementari**, ciascuno robusto a un
sottoinsieme diverso dei problemi. Se più test concordano, il risultato è più affidabile.

---

### TEST 1 — Welch t-test a campione singolo (test primario)

#### Cos'è in parole semplici

Immaginate di avere due liste di numeri: 52 osservazioni settimanali del margine nel
2019 (la nostra "normalità") e 20–27 osservazioni settimanali nel periodo post-shock.
Il Welch t-test a campione singolo risponde alla domanda: **la media della lista
post-shock è abbastanza lontana dalla media del 2019 da non poter essere spiegata
dalla casualità?**

"Abbastanza lontana" dipende da due cose: la distanza effettiva (il delta) e la
dispersione (quanto variano i valori). Se i valori variano molto, serve un delta grande
per concludere che c'è qualcosa di reale. Se variano poco, anche un delta piccolo è
significativo.

#### Perché è il test primario

Il Welch t è il test standard nella letteratura economica per confronti di questo tipo.
Mantenerlo come primario garantisce confrontabilità con altri studi. Usa la variante
"Welch" (invece del t-test classico) che non assume varianze uguali tra i due campioni
— più corretto quando confrontiamo un'intera distribuzione annuale (2019, n=52) con
una finestra di crisi (n=20-27).

#### Il p-value: come si interpreta

Il **p-value** è la probabilità di osservare un risultato così estremo (o più estremo)
*se H₀ fosse vera*. Un p-value di 0.05 significa: "se il margine post-shock fosse
davvero uguale al 2019, avrei solo il 5% di probabilità di ottenere questo risultato
per pura casualità."

Convenzione: p < 0.05 → "statisticamente significativo" (rifiutiamo H₀).

Ma attenzione: un p-value piccolo non dice *quanto è grande* l'effetto, solo che
esiste. Un effetto minuscolo può avere p piccolo con campioni grandi.

#### Risultati

| Evento | Carburante | δ vs 2019 | t-statistic | p-value | Esito |
|---|---|---|---|---|---|
| Ucraina | Benzina | +0.089 EUR/L | 4.490 | 0.0001 | ★★★ RIFIUTA H₀ |
| Ucraina | Diesel | +0.073 EUR/L | 3.531 | 0.0008 | ★★★ RIFIUTA H₀ |
| Iran-Israele | Benzina | +0.079 EUR/L | 20.484 | 0.0000 | ★★★ RIFIUTA H₀ |
| Iran-Israele | Diesel | +0.060 EUR/L | 9.317 | 0.0000 | ★★★ RIFIUTA H₀ |
| Hormuz (prel.) | Benzina | +0.108 EUR/L | 3.270 | 0.0085 | ★★ RIFIUTA H₀ |
| Hormuz (prel.) | Diesel | −0.010 EUR/L | −0.140 | 0.5532 | n.s. non rifiuta |

**Interpretazione:** In tutti i casi non preliminari (Ucraina e Iran-Israele) il margine
post-shock è significativamente superiore al 2019. Il delta è economicamente
rilevante: +0.089 EUR/L per la benzina Ucraina significa che ogni litro costava
circa 9 centesimi in più del normale solo come margine del distributore.

**Il problema di questo test:** come visto nel passo 2, le serie hanno autocorrelazione
forte (DW ≈ 0.3–0.4). Questo gonfia artificialmente la statistica t, perché il test
tratta 25 osservazioni settimanali come se fossero 25 misurazioni indipendenti — ma
se ogni settimana è molto simile alla precedente, le informazioni reali sono molte meno.
Per questo il Welch t non è sufficiente da solo.

---

### TEST 2 — Mann-Whitney U (MW)

#### Cos'è in parole semplici

Il Mann-Whitney non lavora con le medie ma con i **ranghi**: prende tutte le osservazioni
del 2019 (n=52) e tutte quelle post-shock (n=20-27), le mette in ordine dal valore più
basso al più alto, e poi chiede: **le osservazioni post-shock tendono a stare in posizioni
più alte nella classifica rispetto a quelle del 2019?**

Analogia: immaginate due squadre di ciclisti che partecipano alla stessa gara. Non vi
interessa quanto ciascuno è veloce in valore assoluto, solo se i ciclisti di una squadra
arrivano sistematicamente prima degli altri. Il Mann-Whitney fa esattamente questo.

#### Perché è necessario

Il Mann-Whitney non assume che i dati seguano una distribuzione normale (non è
parametrico). Questo è cruciale perché i diagnostici SW hanno mostrato non-normalità
per Ucraina e Hormuz benzina. Un test non parametrico non è "peggiore" del t-test:
è semplicemente più robusto quando i dati non sono normali.

#### Come confronta: post vs distribuzione 2019 intera

In questa pipeline il MW confronta la finestra post-shock con l'intera distribuzione
del 2019 (52 osservazioni), non con la finestra pre-shock dello stesso evento. Perché?

Perché la finestra pre-shock (i mesi prima della guerra, ad esempio) è già potenzialmente
"contaminata": i prezzi cominciavano già a muoversi. Se usassimo quella come riferimento,
staremmo confrontando il post-shock con un pre-shock già anomalo, sottostimando l'effetto.

Usare il 2019 intero come riferimento ci dà una baseline pulita e allineata con H₀:
il margine "normale" è quello del 2019, non quello dell'estate prima della guerra.

#### Statistiche aggiuntive

Il Mann-Whitney produce statistiche descrittive utili:

- **AUC** (Area Under the Curve): probabilità che un'osservazione post-shock scelta
  a caso sia maggiore di una del 2019. AUC = 0.5 = nessuna differenza, AUC = 1.0 =
  tutte le post-shock sono maggiori di tutte le 2019.
- **Hodges-Lehmann (HL)**: stima robusta della differenza tipica tra i due gruppi
  (analoga alla mediana delle differenze). Meno sensibile agli outlier rispetto alla media.
- **Cliff's delta**: effect size tra −1 e +1. Regola empirica: |δ| < 0.147 trascurabile,
  0.147–0.33 piccolo, 0.33–0.474 medio, > 0.474 grande.

#### Risultati

| Evento | Carburante | AUC | Cliff's δ | Magnitudine | p (one-sided) | Esito |
|---|---|---|---|---|---|---|
| Ucraina | Benzina | 0.752 | +0.504 | **grande** | 0.0001 | ★★★ RIFIUTA |
| Ucraina | Diesel | 0.775 | +0.550 | **grande** | 0.0000 | ★★★ RIFIUTA |
| Iran-Israele | Benzina | 1.000 | +1.000 | **grande** | 0.0000 | ★★★ RIFIUTA |
| Iran-Israele | Diesel | 0.940 | +0.881 | **grande** | 0.0000 | ★★★ RIFIUTA |

AUC = 1.000 per Iran-Israele benzina significa che ogni singola settimana del periodo
post-shock ha avuto un margine superiore a ogni singola settimana del 2019. Un segnale
di anomalia di livello assoluto.

---

### TEST 3 — Block Permutation Test (perm)

#### Cos'è in parole semplici

Il permutation test è un approccio radicalmente diverso: invece di appoggiarsi a
formule matematiche, usa la simulazione. L'idea è:

1. Osserviamo la differenza reale (post − pre nel crack spread mediano): chiamiamola D_osservata.
2. Mescoliamo casualmente i dati (pre e post insieme) migliaia di volte e ogni volta
   calcoliamo la differenza: costruiamo così la distribuzione di quello che otterremmo
   *per puro caso*.
3. Il p-value è: in quante simulazioni su 10.000 ottengo una differenza ≥ D_osservata?
   Se solo nel 2% dei casi, p = 0.02.

Analogia: avete due mazzi di carte e volete sapere se uno è "più alto" dell'altro.
Mischiate tutto e ridistribuite a caso 10.000 volte. Quante volte, per puro caso,
un mazzo viene fuori mediamente più alto dell'altro? Se quasi mai, allora la differenza
originale non era dovuta al caso.

#### Perché "a blocchi"

La versione standard del permutation test mescola le singole osservazioni —
ma questo ignora l'autocorrelazione: le settimane adiacenti sono correlate, e
rimescolarle singolarmente spezza quella struttura, rendendo il test troppo ottimistico.

La versione **a blocchi** (block size = 4 settimane ≈ 1 mese) mescola blocchi di 4
settimane consecutive invece di singole osservazioni. In questo modo preserva
la struttura temporale locale nelle permutazioni. È una soluzione parziale
all'autocorrelazione, molto più realistica del test standard.

#### Il problema dello split: τ_price vs τ_margin (e perché sono entrambi implementati)

Il permutation test confronta un gruppo "pre" con un gruppo "post". Ma dove mettiamo
il confine? Ci sono due scelte ragionevoli, e questa pipeline le implementa entrambe:

**Split principale (τ_price):** usa come confine il changepoint del *prezzo* wholesale
stimato nel passo 2. Perché? Perché τ_price è *esogeno* al margine — è stato stimato
guardando solo i prezzi all'ingrosso, non il margine. Non c'è circolarità. La finestra
"pre" termina esattamente quando il prezzo wholesale ha cominciato a muoversi,
garantendo che il gruppo di controllo sia pulito. Questo split entra nella correzione BH.

**Split robustness (τ_margin):** usa come confine il changepoint del *margine* stesso
(stimato con lo stesso metodo brute-force del passo 2, ma sul crack spread invece
che sul prezzo). È endogeno — il confine è scelto guardando la stessa variabile
che stiamo testando — ma cattura esattamente la rottura che ci interessa.
Non entra nella BH, serve come verifica.

**Perché entrambi?** Esiste una tensione metodologica, soprattutto per Ucraina:
τ_price = 3 gennaio 2022, τ_margin = 14 marzo 2022 (gap di 70 giorni). Con lo split
a τ_price, la finestra post include 10 settimane in cui il margine stava ancora salendo
gradualmente — il che diluisce il delta mediano nel test. Con τ_margin come split,
il segnale è molto più netto. Entrambi dicono qualcosa di vero su fenomeni diversi:
*quando si è mosso il prezzo* vs *quando si è rotto il margine*.

#### Risultati

| Evento | Carburante | Split | Δ mediano | p-value | Esito |
|---|---|---|---|---|---|
| Ucraina | Benzina | τ_price (principale) | +0.038 EUR/L | 0.125 | n.s. |
| Ucraina | Benzina | τ_margin (robustness) | +0.095 EUR/L | 0.008 | ★★ |
| Ucraina | Diesel | τ_price (principale) | +0.011 EUR/L | 0.365 | n.s. |
| Ucraina | Diesel | τ_margin (robustness) | +0.060 EUR/L | 0.009 | ★★ |
| Iran-Israele | Benzina | τ_price (principale) | −0.017 EUR/L | 0.913 | n.s. |
| Iran-Israele | Benzina | τ_margin (robustness) | −0.016 EUR/L | 0.930 | n.s. |
| Iran-Israele | Diesel | τ_price (principale) | −0.022 EUR/L | 0.964 | n.s. |
| Iran-Israele | Diesel | τ_margin (robustness) | −0.022 EUR/L | 0.973 | n.s. |

La convergenza tra i due split per Iran-Israele (risultati quasi identici) conferma
che il gap di 7 giorni tra τ_margin e τ_price in quel caso è empiricamente trascurabile.

---

### TEST 4 — HAC Newey-West (HAC)

#### Cos'è in parole semplici

HAC sta per **Heteroscedasticity and Autocorrelation Consistent**. È una tecnica
che prende il test t classico e lo "corregge" per tener conto dell'autocorrelazione
e dell'eteroschedasticità nei residui.

Analogia: il test t classico calcola l'errore standard come se ogni osservazione
fosse indipendente. Se le osservazioni sono correlate (come in una serie settimanale),
l'errore standard viene sottostimato — come se steste misurando l'altezza di mille
persone ma in realtà steste misurando 20 famiglie per 50 volte (misurazioni ridondanti,
non indipendenti). Il Newey-West allarga l'errore standard in modo proporzionale
alla correlazione presente, usando una finestra di 4 lag (4 settimane = 1 mese).

#### Perché è necessario

Con DW = 0.29–0.42, i residui hanno autocorrelazione ρ ≈ 0.85–0.91. Questo significa
che le 25–27 osservazioni post-shock non valgono 25 osservazioni indipendenti,
ma l'equivalente di circa 5–7. Il test t tratta invece tutte come indipendenti,
gonfiando la potenza del test di 3–5 volte. L'HAC corregge questo problema mantenendo
un framework parametrico (più familiare e comparabile con la letteratura).

#### Risultati (split principale τ_price)

| Evento | Carburante | δ HAC | p HAC | Esito |
|---|---|---|---|---|
| Ucraina | Benzina | +0.041 EUR/L | 0.178 | n.s. |
| Ucraina | Diesel | +0.019 EUR/L | 0.496 | n.s. |
| Iran-Israele | Benzina | −0.019 EUR/L | 0.067 | n.s. |
| Iran-Israele | Diesel | −0.027 EUR/L | 0.042 | ★ (negativo) |

Il HAC Iran-Israele diesel con τ_margin: −0.028 EUR/L, p = 0.023 → il margine diesel
è sceso dopo τ_margin, coerentemente con la classificazione "compressione margine".

---

### Come si combinano i quattro test: il "consensus"

L'analisi multi-split mostra, per ogni combinazione evento × carburante × split, quanti
dei quattro test rigettano H₀. La notazione `[n/4 ✓]` indica consensus:

- `[4/4 ✓]` Tutti e quattro i test concordano → risultato molto robusto
- `[3/4 ✓]` Tre su quattro concordano → risultato robusto
- `[2/4 ✓]` Metà concordano → evidenza mista
- `[1/4 –]` Solo uno concorda → risultato debole, cautela
- `[0/4 –]` Nessuno rigetta → nessuna anomalia

Esempio dalla pipeline — Ucraina benzina:

| Split | Consensus |
|---|---|
| shock_hard | `[3/4 ✓]` |
| τ_price | `[1/4 –]` (il consenso si abbassa per il motivo del gap 70gg) |
| τ_margin | `[4/4 ✓]` |
| pre_2019 | `[2/2 ✓]` (solo Welch e MW, no perm/HAC) |

---

### Classificazione finale

Ogni coppia evento × carburante riceve una classificazione basata sull'insieme dei test:

- **Margine anomalo positivo:** il margine post-shock supera la soglia 2σ del 2019
  e almeno un test primario (Welch o MW) rigetta H₀.
- **Compressione margine:** il margine post-shock è inferiore al pre-shock (δ_locale < 0),
  anche se il livello assoluto rimane sopra il 2019.
- **Neutro / trasmissione attesa:** il margine post-shock non supera la soglia 2σ,
  o tutti i test non rigettano H₀.
- **Variazione statistica:** i test non concordano o il segnale è ambiguo.

---

### La finestra pre-shock era già anomala? (δ_pre)

Per ogni evento calcoliamo anche il δ_pre = media(pre-shock) − μ_2019. Se questo
è già > 2σ, significa che il margine era elevato *prima* dello shock — possibile
anticipazione dei mercati o pressioni di costo preesistenti.

Risultati significativi:
- Iran-Israele benzina: δ_pre = +0.086 EUR/L → **pre anomalo** (il margine era già
  alto prima del conflitto). Il margine post è in realtà *sceso* rispetto al pre:
  δ_locale = −0.006 EUR/L.
- Iran-Israele diesel: δ_pre = +0.081 EUR/L → **pre anomalo**. δ_locale = −0.021 EUR/L.

Questo cambia radicalmente l'interpretazione: per Iran-Israele non c'è stata
*espansione* del margine durante la crisi, ma piuttosto il margine era già strutturalmente
elevato nei mesi precedenti, e lo shock lo ha anzi leggermente compresso.

---

### Il windfall: stima dell'extramargine cumulato

Per ogni evento si calcola il **windfall** (extramargine): la somma settimanale di
(margine_t − μ_2019) moltiplicata per i volumi di carburante venduti in Italia
(proxy da dati MISE 2022). Dà un'idea dell'entità economica dell'anomalia.

| Evento | Carburante | Settimane sopra baseline | Windfall lordo (M€) |
|---|---|---|---|
| Ucraina | Benzina | 42/52 | ≈ 649 M€ |
| Ucraina | Diesel | 40/52 | ≈ 1.917 M€ |
| Iran-Israele | Benzina | 43/43 | ≈ 648 M€ |
| Iran-Israele | Diesel | 42/43 | ≈ 1.852 M€ |

**Nota importante:** questi sono extramargini *lordi* calcolati su proxy di volumi e
crack spread. Non sono profitti netti — dal margine lordo vanno sottratti costi
operativi, logistica, struttura della rete distributiva. Sono una stima dell'ordine
di grandezza del fenomeno, non del guadagno effettivo.

---

## PASSO 4 — Evidenza ausiliaria (`04_auxiliary_evidence.py`)

### Tre domande aggiuntive

Il passo 3 ha stabilito se il margine è anomalo rispetto al 2019. Il passo 4 aggiunge
tre domande di contesto: quanto velocemente si trasmettono i prezzi (Granger),
se la trasmissione è strutturalmente asimmetrica (Rockets & Feathers), e se
l'anomalia è specifica all'Italia o comune a tutta l'EU (DiD).

---

### §4.1 — Test di causalità di Granger

#### Cos'è in parole semplici

Il test di Granger risponde a: **sapere il prezzo del Brent di questa settimana aiuta
a prevedere il prezzo alla pompa della prossima settimana?** Se sì, diciamo che il
Brent "Granger-causa" il prezzo alla pompa.

Non è causalità nel senso filosofico — è predittività temporale. Se il Brent di
oggi mi aiuta a prevedere la pompa di domani, e non viceversa, significa che le
informazioni fluiscono in quella direzione.

#### Perché è utile qui

Se il Brent predice la pompa con 1–4 settimane di lag, la trasmissione è veloce.
Un lag breve è ambiguo: potrebbe indicare efficienza del mercato (i prezzi si aggiornano
rapidamente ai costi), ma anche pricing opportunistico (i prezzi salgono immediatamente
quando il wholesale sale). Non ci permette di distinguere le due ipotesi da solo.

#### Risultati

Il test è significativo (p < 0.0001) per tutti i lag da 1 a 8 settimane, sia per
benzina che diesel. F-statistic massima: F=74 per benzina lag-1, F=83 per diesel lag-1.

Questo conferma che il Brent predice fortemente i prezzi alla pompa anche a distanza
di 4 settimane. È coerente sia con mercati efficienti che con pricing anticipatorio.
Per questo il Granger non è classificato come test confirmatory ma esplorativo.

---

### §4.2 — Rockets & Feathers (Asimmetria nella trasmissione)

#### Cos'è in parole semplici

L'effetto "razzi e piume" (Rockets & Feathers) descrive un fenomeno osservato in
molti mercati dei carburanti: **i prezzi alla pompa salgono velocemente quando il
prezzo del Brent sale (razzi), ma scendono lentamente quando il Brent scende (piume)**.

Se questo è vero strutturalmente, i distributori guadagnano di più nei periodi di
discesa rispetto a un mercato perfettamente simmetrico — anche senza fare nulla di
speciale durante le crisi.

#### Come si misura

Si stima un modello OLS separando i movimenti del Brent in rialzi (Δ_up) e ribassi
(Δ_down) e stimando un coefficiente β per ciascuno. Se β_up > β_down, i prezzi
trasmettono i rialzi più velocemente dei ribassi.

Gli errori standard vengono corretti con il metodo Newey-West (stesso HAC del test 4)
perché, come visto, le serie hanno autocorrelazione forte.

#### Risultati

| Carburante | β_up | β_down | R&F index (β_up/β_down) | p asimmetria |
|---|---|---|---|---|
| Benzina | 0.167 | 0.211 | 0.794 | 0.745 n.s. |
| Diesel | 0.240 | 0.183 | 1.314 | 0.757 n.s. |

Il p-value dell'asimmetria è molto alto (non significativo) per entrambi. Questo
significa che i dati non supportano un'asimmetria strutturale statisticamente
distinguibile dal rumore. Il segnale di anomalia del margine non è spiegabile con
un effetto "razzi e piume" preesistente.

---

### §4.3 — Difference-in-Differences (DiD)

#### Cos'è in parole semplici

Il DiD è un metodo per rispondere a: **l'anomalia del margine italiano è specifica
all'Italia, oppure si è verificata allo stesso modo anche in altri paesi EU?**

Funziona così: confrontiamo la variazione del margine italiano con la variazione
del margine di un paese di controllo (Germania e Svezia) durante lo stesso evento.
Se l'Italia ha avuto un salto *aggiuntivo* rispetto alla Germania, quel salto
differenziale non può essere spiegato da un fenomeno comune europeo (es. tutti
i mercati hanno reagito allo stesso modo allo stesso shock).

Formula:
```
δ_DiD = (margine_IT_post − margine_IT_pre) − (margine_DE_post − margine_DE_pre)
```

#### Il Parallel Trends Assumption (PTA): la condizione di validità

Il DiD funziona solo se, in assenza dello shock, Italia e paese di confronto avrebbero
seguito la stessa traiettoria. Questa ipotesi si chiama **Parallel Trends Assumption
(PTA)**.

Come la testiamo? Guardiamo le 8 settimane precedenti allo shock: se le variazioni
settimanali di Italia e paese di controllo erano già divergenti prima dello shock,
il DiD non è interpretabile causalmente.

Il PTA viene testato con una regressione: se il suo p-value è > 0.05, le tendenze
erano parallele e il DiD è valido.

#### Risultati

Tutti i test DiD non risultano significativi (tutti p > 0.05). Questo non significa
che non ci sia anomalia — significa che l'Italia non si è comportata diversamente
da Germania e Svezia durante gli stessi eventi.

Interpretazione: l'anomalia del margine è un fenomeno *europeo*, non specificamente
italiano. Tutti i mercati EU hanno visto margini elevati, probabilmente perché tutti
usano gli stessi futures wholesale come riferimento di costo e tutti hanno reagito
agli stessi shock di mercato.

---

## PASSO 5 — Correzione per test multipli (`05_global_corrections.py`)

### Il problema dei test multipli: l'esempio dei dadi

Immaginate di tirare un dado a sei facce 20 volte e scommettere ogni volta che esce 6.
Naturalmente, in media esce 6 circa 3 volte su 20 — per pura casualità. Se dopo 20
tiri avete vinto 3 volte e concludete "il dado è truccato", state sbagliando: è
esattamente quello che ci aspettavamo per caso.

Lo stesso problema esiste quando eseguiamo molti test statistici: se usiamo soglia
p < 0.05 per ciascuno, su 24 test indipendenti ci aspettiamo circa 1.2 rigetti falsi
*per pura casualità*. Dobbiamo correggere.

### Cos'è la correzione Benjamini-Hochberg (BH)

La correzione **Benjamini-Hochberg (1995)** è un metodo che controlla il **False
Discovery Rate (FDR)**: la proporzione attesa di rigetti falsi tra tutti i rigetti.
A differenza della più conservativa correzione di Bonferroni (che controlla la
probabilità di *anche un solo* falso positivo), la BH permette di trovare più risultati
veri mantenendo il FDR ≤ 5%.

Come funziona:
1. Ordina tutti i p-value dal più piccolo al più grande.
2. Assegna a ciascuno un rango k (da 1 a n).
3. Rifiuta H₀ per tutti i test con p ≤ (k/n) × α.

Questo crea una soglia adattiva: i test con p-value bassissimi vengono confrontati
con una soglia più permissiva, quelli con p-value alti con una soglia più severa.

### Quale famiglia di test entra nella BH

Entrano nella BH solo i test **confirmatory**: quelli che testano direttamente H₀
"il margine non è anomalo". I test esplorativi (Granger, Rockets & Feathers) rispondono
a domande diverse e non devono essere mescolati.

| Categoria | Script | N test | Entra nella BH? |
|---|---|---|---|
| Welch t (primario) | 03 | 4 | ✓ Sì |
| Mann-Whitney (vs 2019) | 03 | 4 | ✓ Sì |
| Block permutation (split τ_price) | 03 | 4 | ✓ Sì |
| HAC Newey-West (split τ_price) | 03 | 4 | ✓ Sì |
| DiD IT vs DE e SE | 04 | 8 | ✓ Sì |
| Block perm (split τ_margin, robustness) | 03 | 4 | ✗ No (check) |
| HAC (split τ_margin, robustness) | 03 | 4 | ✗ No (check) |
| Granger, Rockets & Feathers | 04 | — | ✗ No (esplorativi) |
| Hormuz (tutti) | 03/04 | — | ✗ No (preliminare) |
| **Totale famiglia BH** | | **24** | |

### Risultati finali

Dalla famiglia di 24 test, la BH globale rigetta H₀ per 8 test:

| Test | Evento | Carburante | p nominale | p aggiustato |
|---|---|---|---|---|
| Mann-Whitney | Ucraina | Diesel | 0.0000 | 0.0000 |
| Mann-Whitney | Iran-Israele | Benzina | 0.0000 | 0.0000 |
| Mann-Whitney | Iran-Israele | Diesel | 0.0000 | 0.0000 |
| Welch t | Iran-Israele | Benzina | 0.0000 | 0.0000 |
| Welch t | Iran-Israele | Diesel | 0.0000 | 0.0000 |
| Welch t | Ucraina | Benzina | 0.0001 | 0.0003 |
| Mann-Whitney | Ucraina | Benzina | 0.0001 | 0.0003 |
| Welch t | Ucraina | Diesel | 0.0008 | 0.0024 |

**Sommario per famiglia di test:**
- Welch t: 4/4 rigettati (FDR 5%)
- Mann-Whitney: 4/4 rigettati (FDR 5%)
- Block permutation (principale): 0/4 rigettati
- HAC (principale): 0/4 rigettati
- DiD: 0/8 rigettati

### Come interpretare questo quadro

**I test di livello assoluto (Welch, MW vs 2019) rigettano tutti.** Questo dice che il
margine nel periodo post-shock è sistematicamente più alto del 2019, in tutti e 4 i
casi non preliminari, con certezza statistica molto alta anche dopo correzione BH.

**I test di salto locale (perm, HAC con split τ_price) non rigettano.** Questo non
contraddice il risultato precedente: misura una domanda *diversa*. Il permutation test
con split τ_price chiede: "il margine ha fatto un salto brusco nel periodo immediatamente
successivo al changepoint del prezzo?" La risposta è: no per Ucraina (il margine si è
alzato gradualmente nei 70 giorni successivi a τ_price), e no per Iran-Israele (il
margine era già alto prima dello shock e non è ulteriormente salito dopo).

**I test di robustness (split τ_margin) concordano con perm/HAC per Iran-Israele,
ma danno segnale per Ucraina.** Questo conferma che per Ucraina c'è un salto reale del
margine, ma avviene con un ritardo di 70 giorni rispetto al segnale di prezzo —
il che è coerente con trasmissione graduale dei costi piuttosto che con un salto
istantaneo speculativo.

**Il DiD non rigetta.** L'anomalia non è specifica all'Italia: tutti i mercati EU
hanno seguito la stessa dinamica.

---

## Analisi annuale: "alla fine dell'anno si sono guadagnati uguale?"

Oltre all'analisi per evento, la pipeline calcola per ogni anno dal 2019 al 2026
il margine medio e il windfall netto rispetto al 2019. Risponde a una domanda
diversa: anche se il margine è stato volatile, *nell'intero anno* quanto si è
guadagnato in più?

Benzina (extramargine annuo rispetto al 2019, in milioni di euro):

| Anno | Anomalo (2σ)? | MW vs 2019 | Windfall netto M€ |
|---|---|---|---|
| 2019 | no | — | −0 |
| 2020 | no | p=0.0001 | +340 |
| 2021 | no | p=0.0076 | +126 |
| 2022 | **Sì** | p=0.0000 | +807 |
| 2023 | **Sì** | p=0.0000 | +633 |
| 2024 | **Sì** | p=0.0000 | +622 |
| 2025 | **Sì** | p=0.0000 | +818 |
| 2026 (parz.) | **Sì** | p=0.0000 | +288 |

I margini sono rimasti strutturalmente sopra il 2019 per tutti gli anni dal 2022 in poi,
anche pesando i periodi di compressione. Non si tratta di spike temporanei ma di un
livello più alto stabilizzato.

---

## Limitazioni metodologiche: cosa questa analisi non può dire

### 1. Il crack spread è una proxy, non il margine reale

Il "margine" calcolato è la differenza tra prezzo alla pompa e futures wholesale
Amsterdam-Rotterdam-Anversa. I distributori italiani comprano realmente su mercati
CIF-Genova o con contratti bilaterali a prezzi diversi. Se il differenziale ARA-Genova
è cambiato sistematicamente durante le crisi, una parte del "margine anomalo" potrebbe
riflettere questo, non un guadagno aggiuntivo del distributore.

### 2. Causalità vs correlazione

I test statistici descrivono pattern, non cause. "Margine anomalo positivo" è coerente
con diversi meccanismi: comportamento opportunistico, effetti FIFO/LIFO sull'inventario
(se hai comprato il carburante quando costava meno e lo vendi al prezzo di oggi),
risk premium razionale (i distributori possono voler coprire il rischio di future
perdite in un ambiente volatile), costi operativi aumentati per ragioni indipendenti.
Separare queste cause richiederebbe dati di inventario e costi interni non pubblici.

### 3. La baseline 2019 è una scelta

Abbiamo usato il 2019 come anno "normale". Se usassimo il 2021 come baseline
(anno più vicino alla crisi, ma già parzialmente influenzato dalla ripresa post-COVID),
la soglia 2σ sarebbe più alta (0.048 vs 0.038 EUR/L per la benzina) e meno casi
risulterebbero anomali. L'analisi di sensitivity è inclusa nella pipeline
(data/baseline_sensitivity.csv) e riporta entrambi i valori.

### 4. Hormuz è preliminare

Al momento dell'analisi, solo 7 settimane di dati post-shock Hormuz erano disponibili.
I risultati per quel evento vanno trattati come direzioni indicative, non come
conclusioni robuste.

### 5. Il DiD presuppone paesi comparabili

Germania e Svezia non sono sostituti perfetti dell'Italia come gruppo di controllo.
Hanno strutture distributive diverse, mix energetico diverso, regolamentazione diversa.
Il DiD non rigettare non garantisce che i mercati siano identici — solo che la
*variazione* del margine durante gli eventi è stata simile.

---

## Glossario dei termini tecnici

| Termine | Significato semplice |
|---|---|
| **Crack spread** | Differenza tra prezzo alla pompa (netto tasse) e costo wholesale: proxy del margine lordo |
| **Baseline** | Anno di riferimento "normale" (2019) usato come confronto |
| **H₀ / H₁** | Ipotesi nulla (niente di speciale) / alternativa (qualcosa è cambiato) |
| **p-value** | Probabilità di ottenere un risultato così estremo per puro caso, assumendo H₀ vera |
| **Deviazione standard (σ)** | Misura di quanto variano normalmente i dati |
| **Autocorrelazione** | Tendenza di una serie a essere simile a sé stessa la settimana precedente |
| **Eteroschedasticità** | Varianza dei dati che cambia nel tempo |
| **Changepoint (τ)** | Data in cui la traiettoria di una serie ha cambiato direzione strutturalmente |
| **MCMC** | Metodo di simulazione per stimare distribuzioni di probabilità |
| **Rhat** | Indicatore di convergenza delle simulazioni MCMC (buono se vicino a 1.00) |
| **Test parametrico** | Test che assume una forma specifica della distribuzione (es. normale) |
| **Test non parametrico** | Test che non fa assunzioni sulla distribuzione (es. Mann-Whitney) |
| **Benjamini-Hochberg (BH)** | Correzione statistica per quando si fanno molti test insieme |
| **FDR** | False Discovery Rate: proporzione attesa di risultati falsi tra quelli rigettati |
| **DiD** | Difference-in-Differences: confronto tra paese trattato e gruppo di controllo |
| **PTA** | Parallel Trends Assumption: condizione di validità del DiD |
| **Granger** | Test che verifica se una serie predice un'altra nel tempo |
| **HAC** | Correzione degli errori standard per autocorrelazione e eteroschedasticità |
| **Block permutation** | Test di simulazione che preserva la struttura temporale locale |
| **τ_price** | Data del changepoint stimato sui prezzi wholesale |
| **τ_margin** | Data del changepoint stimato sul crack spread (margine) |
| **Cliff's delta** | Effect size ordinale: quanto sono spostati i valori di un gruppo rispetto all'altro |
| **Hodges-Lehmann** | Stima robusta della differenza tipica tra due gruppi (alternativa alla media) |
| **Windfall** | Extramargine cumulato sopra la baseline, moltiplicato per i volumi venduti |

---