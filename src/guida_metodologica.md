# Guida Metodologica Completa
## Come funziona la pipeline di analisi dei margini sui carburanti italiani
### Versione aggiornata post-run 27 aprile 2026

---

> **Come leggere questo documento.** Ogni sezione spiega prima *cosa* fa un passaggio,
> poi *perché* è stato scelto, poi *come si interpreta* il risultato, poi *cosa non funziona*
> o dove l'analisi è migliorabile. I numeri provengono dall'output reale della pipeline.
> I box "⚠ PROBLEMA METODOLOGICO" segnalano questioni aperte.

---

## Il problema che vogliamo risolvere

Quando scoppia una crisi energetica i prezzi alla pompa salgono. È normale: il costo delle
materie prime aumenta e i distributori lo trasferiscono ai consumatori.

La domanda è più precisa: **i distributori italiani hanno trasferito solo il maggior costo,
oppure ne hanno approfittato per allargare il proprio margine rispetto ai periodi normali?**

Guardiamo la *differenza* tra prezzo alla pompa (senza tasse) e costo del carburante sul
mercato all'ingrosso europeo. Questa differenza si chiama **crack spread** ed è la proxy
del margine lordo del distributore.

Se il crack spread durante la crisi è simile a quello del 2019 (anno normale pre-COVID),
i distributori hanno fatto pass-through corretto. Se è significativamente più alto, c'è
un'anomalia che merita spiegazione.

---

## La struttura della pipeline (v2)

```
01_data_pipeline.py        → Raccoglie e pulisce i dati (381 settimane, 2019-2026)
02_changepoint.py          → Quando si è rotta la dinamica dei prezzi? (MCMC bayesiano)
03_margin_hypothesis.py    → Il margine è anomalo rispetto al 2019? (BH-A: 16 test)
04_auxiliary_evidence.py   → L'anomalia è specifica all'Italia? (BH-B DiD: 8 test)
05_global_corrections.py   → Quanti risultati reggono? (BH globale 24 test: 16+8)
06_distribution_check.py   → n_eff, Andrews BW, diagnostica distribuzione
07_preshock_anomaly.py     → Anomalia strutturale pre-shock Iran-Israele (Bai-Perron)
```

**Architettura epistemica v2:**
- **BH-A (16 test):** confirmativi H₀: μ_post = μ₂₀₁₉, split esogeni (shock_hard + τ_price)
- **BH-B (8 test):** ausiliari DiD H₀: δ_DiD = 0 — specificità italiana
- **Esplorativi [no BH]:** τ_margin, block perm, HAC, n_eff — rispondono a domande diverse
- **Pre-shock [no BH]:** anomalia strutturale 2025-H1 (script 07)

---

## PASSO 1 — Raccolta dati

### Cosa fa

Raccoglie da fonti pubbliche tre serie settimanali (381 settimane, gen 2019 – apr 2026):
- **Brent crude EUR/barile** — Yahoo Finance, convertito con EUR/USD settimanale
- **Prezzi pompa italiani senza tasse** — EU Weekly Oil Bulletin, foglio "Prices wo taxes"
- **Futures wholesale europei** — Eurobob ARA (benzina) e Gas Oil ICE (diesel), CSV da Investing.com

### Perché queste scelte

**Prezzi senza tasse:** le accise italiane sono fisse. Includere le tasse mescolerebbe una
componente invariante con quella che riflette il comportamento del distributore.

**Eurobob/Gas Oil invece del Brent:** il Brent è il prezzo del greggio. Tra greggio e pompa
ci sono costi di raffinazione non banali. Eurobob e Gas Oil ICE sono i prezzi effettivi
che i distributori pagano per prodotto già raffinato sul mercato ARA.

**Baseline 2019:** mercato maturo, niente COVID, Brent stabile 60–70 $/bbl. È il
"termometro della normalità" pre-crisi.

### Il crack spread: il margine lordo analizzato

```
crack spread = prezzo pompa (netto tasse) − prezzo wholesale ARA
```

Baseline 2019: **benzina μ=0.168 EUR/L (σ=0.019), diesel μ=0.149 EUR/L (σ=0.018)**.
Questi includono: trasporto, stoccaggio, margine retail, gestione rete distributiva.

> ⚠ **LIMITAZIONE STRUTTURALE:** il wholesale di riferimento è ARA (Amsterdam-Rotterdam-
> Antwerp). I distributori italiani importano via Mediterraneo (Genova, Livorno). Il
> differenziale ARA-Med può variare sistematicamente durante le crisi (rerouting navale,
> costi assicurativi). Una parte del "margine anomalo" potrebbe riflettere questo.

---

## PASSO 2 — Changepoint bayesiano

### Cosa fa

Individua **quando** la traiettoria dei log-prezzi ha subito una rottura strutturale,
senza imporre a priori la data dell'evento geopolitico. Usa un modello piecewise-lineare
con likelihood StudentT (robustezza alle code pesanti dei prezzi energetici).

Il **lag D = τ − shock_date** misura l'anticipo (D < 0) o ritardo (D > 0) del
changepoint rispetto allo shock.

Come effetto collaterale produce i diagnostici OLS (DW, SW, BP) che guidano la scelta
dei test nel passo 3.

### Analogia

Immaginate il prezzo del Brent come una linea che sale e scende. Il changepoint è il
momento in cui quella linea ha cambiato *direzione sistematica* — non una piccola
oscillazione, ma un cambio di pendenza permanente. Il modello divide la serie in due
tratti e trova il punto di divisione che spiega meglio i dati.

### Perché bayesiano

Un approccio banale imporrebbe la data dell'evento come soglia. I mercati finanziari
sono *forward-looking*: anticipano gli eventi. Il modello lascia che i *dati* determinino
il punto di rottura, producendo un'intera distribuzione posteriore su τ (non solo un
punto puntuale).

### Risultati — Table 1

| Evento | Serie | τ̂ | Lag D | Rhat | Note |
|---|---|---|---|---|---|
| Ucraina | Brent | 2021-12-13 | −73 gg | **1.162 ⚠** | MCMC non converge |
| Ucraina | Benzina | 2022-01-03 | −52 gg | 1.001 ✓ | ν=1.47 (code molto pesanti) |
| Ucraina | Diesel | 2022-01-03 | −52 gg | 1.004 ✓ | |
| Iran-Israele | Brent | 2025-04-28 | −46 gg | 1.002 ✓ | |
| Iran-Israele | Benzina | 2025-04-28 | −46 gg | 1.002 ✓ | |
| Iran-Israele | Diesel | 2025-05-05 | −39 gg | 1.001 ✓ | |
| Hormuz | Brent | 2026-02-16 | −12 gg | 1.002 ✓ | Trasmissione rapida |
| Hormuz | Benzina | 2026-03-02 | +2 gg | 1.002 ✓ | Quasi sincrono |
| Hormuz | Diesel | 2026-02-23 | −5 gg | 1.001 ✓ | |

**Interpretazione dei lag:** per Ucraina e Iran-Israele il mercato aveva anticipato lo
shock di 39–52 giorni (forward-looking). Per Hormuz (crisi più improvvisa) il changepoint
coincide quasi esattamente con l'evento.

> ⚠ **BRENT UCRAINA (Rhat=1.162):** la catena MCMC non ha convergito. τ_Brent-Ucraina
> non è affidabile. Importante: questo τ non viene usato nei test sul margine — lì si usa
> il τ di Benzina e Diesel, che convergono. Ma andrebbe corretto con un re-run dedicato.

### I diagnostici che motivano i test successivi

**Durbin-Watson (DW)** — misura l'autocorrelazione. Tutti e 9 i casi mostrano DW < 1.5,
indicando autocorrelazione forte. I valori DW 0.29–0.42 corrispondono a ρ̂ ≈ 0.85–0.90.

> ⚠ **IMPLICAZIONE CRITICA dell'autocorrelazione:** con ρ≈0.85, il numero effettivo
> di osservazioni indipendenti in una finestra di 25 settimane è:
> n_eff ≈ 25 × (1−0.85)/(1+0.85) ≈ **2 osservazioni**.
> I test parametrici che trattano 25 osservazioni come 25 indipendenti
> sovrastimano la certezza statistica di circa 3–4 volte.

**Shapiro-Wilk (SW)** — non-normalità rilevata per Ucraina benzina/diesel (p≈0) e
Hormuz benzina (p=0.000). Questo motiva i test non parametrici (Mann-Whitney).

**Breusch-Pagan (BP)** — eteroschedasticità per Iran diesel (p=0.021) e Hormuz benzina
(p=0.010). Gestita parzialmente da Welch t e HAC.

---

## PASSO 3 — Test sull'anomalia del margine

### L'ipotesi nulla

**H₀:** Il margine lordo medio nel periodo post-shock è uguale alla media del 2019.
**H₁:** Il margine lordo medio nel periodo post-shock è superiore alla media del 2019.

Il test è **unilaterale superiore**: ci interessa solo se il margine è salito.

### La soglia di anomalia: ±2σ del 2019

La soglia è posta a 2σ sopra la media del 2019: in una distribuzione normale, circa
il 95% dei valori cade entro ±2σ. Se il margine post-shock supera μ₂₀₁₉ + 2σ, quel
livello sarebbe stato eccezionale anche in un anno normale.

Soglie: **benzina 0.038 EUR/L, diesel 0.037 EUR/L sopra la media 2019**.

### L'architettura multi-split

Una novità importante rispetto alle analisi standard: ogni test è eseguito in parallelo
su tre possibili punti di divisione pre/post.

| Split | Quando | Perché | Problema |
|---|---|---|---|
| shock_hard | Data geopolitica fissa | Confrontabilità con letteratura | La data fissa può non coincidere con il vero cambio di regime |
| τ_price | Changepoint MCMC del prezzo wholesale | Esogeno al margine — nessuna circolarità | Dipende dall'MCMC |
| τ_margin | Changepoint Bai-Perron del margine | Cattura la rottura effettiva del crack spread | **ENDOGENO — CIRCOLARE** |

> ⚠ **PROBLEMA CRITICO — τ_margin nella BH confirmatory:**
> τ_margin è stimato come il punto che massimizza la differenza nel margine tra due
> finestre. Poi si testa se quella differenza è significativa. Questo è **data-snooping**:
> cerchi il posto più "diverso" e poi chiedi "è davvero diverso?". La risposta è quasi
> sempre sì, anche con dati casuali. I p-value prodotti da test su split τ_margin non
> sono validi sotto H₀ e non dovrebbero entrare nella correzione BH.

### Le due domande diverse mescolate nella pipeline

La batteria di test risponde a due domande distinte che andrebbero tenute separate:

**Domanda A — Livello assoluto:** il margine post-shock è più alto del 2019?
→ Welch t-test (H₀: μ_post = μ₂₀₁₉) e Mann-Whitney (post vs distribuzione 2019)

**Domanda B — Salto locale:** il margine ha fatto un salto brusco nel punto di split?
→ Block permutation (H₀: μ_post = μ_pre) e HAC Newey-West (H₀: μ_post = μ_pre)

La pipeline combina le risposte a entrambe le domande nella stessa correzione BH, il che
rende difficile capire quanti e quali rigetti si riferiscono a quale domanda.

---

### TEST 1 — Welch t-test a campione singolo (test primario)

Confronta la media del margine post-shock con la media fissa del 2019. È il test
standard della letteratura economica per questo tipo di confronto.

**Risultati (split τ_price, principale non-endogeno):**

| Evento | Carburante | δ_vs_2019 | δ_local | pre_anom? | p |
|---|---|---|---|---|---|
| Ucraina | Benzina | **+0.074 EUR/L** | +0.041 | no | 0.0000 *** |
| Ucraina | Diesel | **+0.056 EUR/L** | +0.019 | ≈sì | 0.0009 *** |
| Iran-Is. | Benzina | **+0.076 EUR/L** | **−0.019** | **SÌ** | 0.0000 *** |
| Iran-Is. | Diesel | **+0.061 EUR/L** | **−0.027** | **SÌ** | 0.0000 *** |

**Lettura del δ_local negativo per Iran-Israele:** il Welch rigetta H₀ nel senso che
μ_post > μ₂₀₁₉. Ma il periodo pre-shock era già anomalamente elevato (δ_pre benzina
+0.095 EUR/L, diesel +0.088 EUR/L). Il margine è sceso durante la crisi rispetto al
pre-shock. Questi non sono risultati contraddittori: uno misura il livello rispetto al
2019, l'altro misura il cambiamento durante l'evento.

> ⚠ **OVERFITTING dell'evidenza con autocorrelazione:** con ρ≈0.85 e n_post=20-27,
> il Welch t vede ~2 osservazioni indipendenti ma calcola la statistica come se ne
> avesse 25. La statistica t di 4.49 per Ucraina benzina è gonfiata di circa 3.5×;
> quella "reale" sarebbe ~1.3 (non significativa). I rigetti formali sono tecnicamente
> corretti ma la certezza statistica è sopravvalutata.

---

### TEST 2 — Mann-Whitney U

Confronto su ranghi: testa se le osservazioni post-shock tendono sistematicamente
a stare più in alto nella classifica rispetto alle 52 settimane del 2019. Non assume
normalità né varianze uguali.

**Risultati:**

| Evento | Carburante | AUC | Cliff's δ | Magnit. | p |
|---|---|---|---|---|---|
| Ucraina | Benzina | 0.752 | +0.504 | **grande** | 0.0001 *** |
| Ucraina | Diesel | 0.775 | +0.550 | **grande** | 0.0000 *** |
| Iran-Is. | Benzina | **1.000** | **+1.000** | **grande** | 0.0000 *** |
| Iran-Is. | Diesel | 0.940 | +0.881 | **grande** | 0.0000 *** |

AUC=1.000 per Iran-Israele benzina: ogni singola settimana post-shock aveva margine
superiore a ogni singola settimana del 2019. Questo conferma che il LIVELLO è molto
sopra il 2019, ma è coerente con il fatto che fosse già elevato prima dello shock.

Il MW non dipende dalla scelta dello split perché confronta sempre post-shock vs
2019 fisso — questa è la sua forza rispetto al permutation test.

---

### TEST 3 — Block Permutation (block_size=4 settimane)

Mescola casualmente blocchi di 4 settimane consecutive (non osservazioni singole)
per preservare l'autocorrelazione locale. Testa se il salto mediano pre/post è
spiegabile dalla casualità.

**Risultati split τ_price (principale, non endogeno):**

| Evento | Carburante | Δ mediano | p |
|---|---|---|---|
| Ucraina | Benzina | +0.038 | 0.125 **n.s.** |
| Ucraina | Diesel | +0.011 | 0.352 **n.s.** |
| Iran-Is. | Benzina | −0.019 | 0.915 **n.s.** |
| Iran-Is. | Diesel | −0.022 | 0.963 **n.s.** |

**Perché n.s. per Ucraina?** La finestra pre termina al τ_price = 3 gennaio 2022,
ma il margine ha risposto al segnale di prezzo con un ritardo di 70 giorni (τ_margin
= 14 marzo 2022). Le 10 settimane di transizione graduale del margine finiscono nel
"post" e diluiscono il delta mediano. Con split τ_margin (robustness, non in BH):
Ucraina benzina p=0.009, diesel p=0.006 — segnale netto. Questo racconta una storia
coerente: il margine è salito per Ucraina, ma gradualmente e con ritardo.

**Perché n.s. per Iran-Israele (qualunque split)?** Il δ_local è già negativo:
il margine era alto prima dello shock e ha compresso durante. Il permutation test
non può rigettare su un salto negativo.

---

### TEST 4 — HAC Newey-West (maxlags=4)

Mantiene la struttura OLS parametrica ma corregge gli errori standard per
autocorrelazione ed eteroschedasticità. Complementare al permutation test:
dove uno fallisce (potenza), l'altro può compensare.

**Risultati split τ_price:**

| Evento | Carburante | δ_HAC | p |
|---|---|---|---|
| Ucraina | Benzina | +0.041 | 0.178 n.s. |
| Ucraina | Diesel | +0.019 | 0.496 n.s. |
| Iran-Is. | Benzina | −0.019 | 0.067 n.s. |
| Iran-Is. | Diesel | −0.027 | 0.042 * (BH adj. 0.074 → non rigettato) |

> ⚠ **maxlags=4 INSUFFICIENTE per ρ≈0.85:** la correlazione al lag 4 è 0.85⁴ ≈ 0.52.
> Ancora molto alta. La Newey-West dovrebbe usare la selezione automatica del bandwidth
> (Andrews 1991) che per ρ=0.85 suggerirebbe circa 12-15 lag. Con maxlags=4 i p-value
> HAC sottostimano ancora l'incertezza.

---

### Come si combinano i quattro test

Per ogni coppia evento × carburante × split, la pipeline conta quanti test su 4 rigettano H₀:

| Evento | Carb. | Split | Welch | MW | Perm | HAC | Consensus |
|---|---|---|---|---|---|---|---|
| Ucraina | Benzina | τ_price | ✓ | ✓ | ✗ | ✗ | 2/4 |
| Ucraina | Benzina | τ_margin (⚠) | ✓ | ✓ | ✓ | ✓ | 4/4 |
| Ucraina | Diesel | τ_price | ✓ | ✓ | ✗ | ✗ | 2/4 |
| Iran-Is. | Benzina | qualsiasi | ✓ | ✓ | ✗ | ✗ | 2/4 |
| Iran-Is. | Diesel | qualsiasi | ✓ | ✓ | ✗ | ✗ | 2/4 |

Il consensus 4/4 di Ucraina-τ_margin va letto con cautela (split endogeno, §2.1).
Il consensus 2/4 con τ_price è genuino ma moderato: i due test sulla domanda "A"
(livello assoluto vs 2019) concordano, quelli sulla domanda "B" (salto locale) no.

---

### Il τ_margin e la sua relazione con τ_price

| Evento | Carburante | τ_price | τ_margin | Lag | Interpretazione |
|---|---|---|---|---|---|
| Ucraina | Benzina | 2022-01-03 | 2022-03-14 | +70 gg | **REATTIVO** — cost pass-through graduale |
| Ucraina | Diesel | 2022-01-03 | 2022-03-14 | +70 gg | **REATTIVO** |
| Iran-Is. | Benzina | 2025-04-28 | 2025-04-21 | −7 gg | **SINCRONO** — margine e prezzo si muovono insieme |
| Iran-Is. | Diesel | 2025-05-05 | 2025-06-02 | +28 gg | **REATTIVO** |
| Hormuz | Benzina | 2026-03-02 | 2025-11-24 | **−98 gg** | **ANTICIPATORIO** ⚠ |
| Hormuz | Diesel | 2026-02-23 | 2026-01-05 | **−49 gg** | **ANTICIPATORIO** ⚠ |

Il segnale anticipatorio di Hormuz è potenzialmente il più interessante della pipeline:
il margine si è espanso 3 mesi prima dello shock di Hormuz e prima che il prezzo
wholesale salisse. Ma con soli 7 dati post-shock e un Bai-Perron su finestra breve,
l'instabilità della stima è alta. **Da verificare con dati aggiuntivi.**

---

### Classificazione finale (v2)

| Evento | Carburante | δ_vs_2019 | δ_local | pre_anom | n_eff | BH-A | Classificazione |
|---|---|---|---|---|---|---|---|
| Ucraina | Benzina | **+0.089/+0.074** | +0.058/+0.041 | no | 4.0/5.4 ⚠ | ✓ | **Confermato** (cautela n_eff) |
| Ucraina | Diesel | **+0.073/+0.056** | +0.047/+0.019 | no | 8.0/10.4 ⚠ | ✓ | **Confermato** |
| Iran-Is. | Benzina | **+0.079/+0.076** | **−0.006/−0.019** | **SÌ** | 6.3/7.5 ⚠ | ✓ | **Confermato (livello, non causato da shock)** |
| Iran-Is. | Diesel | **+0.060/+0.061** | **−0.021/−0.027** | **SÌ** | 17.5/14.2 | ✓ | **Confermato (livello, non causato da shock)** |
| Hormuz | Benzina | **+0.108** | +0.008 | **SÌ** | 3.1 ⚠ | ✗ prel. | **Inconclusivo** ⚠ |
| Hormuz | Diesel | −0.010 | −0.083 | **SÌ** | 1.3 ⚠ | ✗ prel. | **Neutro** ⚠ |

*(Doppio valore = shock_hard / τ_price)*

**Nota Iran-Israele:** script 07 mostra che l'anomalia strutturale è iniziata a luglio 2024
(break Bai-Perron sulla serie benzina) e intorno a ottobre 2023 per il diesel. Lo shock
ha semmai compresso il margine (δ_local < 0). La classificazione correta è quindi
"**Anomalia strutturale pre-esistente**", non "compressione da shock".

**Nota Hormuz benzina:** il δ_vs_2019=+0.108 (z=5.5) supera ampiamente la soglia 2σ,
ma con n_eff=3.1 e soli 12 post-shock il test non è informativo. Da aggiornare.

---

## PASSO 4 — Evidenza ausiliaria

### §4.1 Granger: velocità di trasmissione

Il test di Granger misura se sapere il Brent di questa settimana aiuta a prevedere
il prezzo alla pompa della prossima. Risultati v2:

| Carburante | Lag 1w | Lag 2w | Lag 3w | Lag 4w |
|---|---|---|---|---|
| Benzina | F=58.2 p<0.001 | F=37.3 p<0.001 | F=30.3 p<0.001 | F=22.9 p<0.001 |
| Diesel | F=42.2 p<0.001 | F=35.3 p<0.001 | F=26.3 p<0.001 | F=20.9 p<0.001 |

Il Brent predice i prezzi pompa con trasmissione molto rapida. **Esplorativo — non entra nella BH.**

### §4.2 Rockets & Feathers: asimmetria strutturale

| Carburante | β_up | β_down | R&F index | p asimmetria |
|---|---|---|---|---|
| Benzina | +0.0039 | +0.0022 | 1.765 | 0.239 n.s. |
| Diesel | +0.0056 | +0.0017 | 3.324 | **0.091 *** |

Benzina: nessuna asimmetria strutturale. Diesel: R&F index = 3.3 (le salite sono 3×
più rapide dei ribassi), con p=0.091 quasi-significativo. Questo è un aggiornamento
rispetto a v1 (diesel p=0.757) — il segnale è cresciuto con più dati. **Esplorativo.**

### §4.3 Difference-in-Differences: l'anomalia è italiana?

> ⚠ **BUG v2 — FILE SBAGLIATO:** lo script 05 carica `auxiliary_pvalues.csv` (v1, p-value 0.2–0.8)
> invece di `did_results_v2.csv` (v2, p≈0 per 7/8 test). La conclusione "0/8 rigettati"
> nel sommario del run è basata su dati obsoleti. I risultati corretti sono quelli di script 04.

**Risultati v2 da `did_results_v2.csv` (fonte corretta):**

| Evento | Paese | Carb. | δ_DiD (EUR/L) | p | PTA | Interpretazione |
|---|---|---|---|---|---|---|
| Ucraina | Germania | Benzina | **+0.184** | 0.0000*** | **VIOLATA** | IT > DE: a favore specificità IT |
| Ucraina | Germania | Diesel | **−0.064** | 0.0019** | valida | IT < DE: contro specificità IT |
| Ucraina | Svezia | Benzina | **+0.202** | 0.0000*** | **VIOLATA** | IT > SE: a favore specificità IT |
| Ucraina | Svezia | Diesel | +0.002 | 0.928 n.s. | **VIOLATA** | n.s. |
| Iran-Is. | Germania | Benzina | **−0.118** | 0.0000*** | **VIOLATA** | IT < DE: contro specificità IT |
| Iran-Is. | Germania | Diesel | **−0.119** | 0.0000*** | valida | IT < DE: contro specificità IT |
| Iran-Is. | Svezia | Benzina | **−0.118** | 0.0000*** | **VIOLATA** | IT < SE: contro specificità IT |
| Iran-Is. | Svezia | Diesel | **−0.119** | 0.0000*** | valida | IT < SE: contro specificità IT |

**Osservazioni critiche v2:**

1. **PTA violata in 5/8 casi.** Quando il pre-trend non è parallelo, il DiD non è
   un confronto causale valido. Solo i 3 casi con PTA valida (Ucraina DE diesel,
   Iran-Is. DE diesel, Iran-Is. SE diesel) sono interpretabili.

2. **δ molto diversi da v1** (es. Ucraina DE benzina: v1=−0.024, v2=+0.184).
   Questo salto è probabilmente dovuto a un cambio di finestra temporale o di
   definizione del margine tra v1 e v2 — va investigato.

3. **Conclusione provvisoria con PTA validi:** Ucraina diesel ← IT inferiore a
   Germania (p=0.002), né a favore né contro in modo netto. Iran-Is. diesel ←
   IT inferiore a entrambi i controlli (p≈0). Evidenza complessiva CONTRO specificità italiana.

**Script 04 dice 7/8 rigettati a α=0.05.** Ma molti rigetti corrispondono a δ negativi
(IT < controllo): rigettare H₀_DiD=0 con δ<0 significa evidenza CONTRO opportunismo IT,
non a favore. La BH-B richiede interpretazione direzionale.

---

## PASSO 5 — Correzione per test multipli (v2: architettura pulita)

### Il cambiamento fondamentale di v2

In v1 la famiglia BH mescolava 56 test con ipotesi diverse (H₀ sul livello assoluto e
H₀ sul salto locale) e includeva τ_margin (endogeno). In v2 la famiglia è separata in
due sotto-famiglie pulite:

| Famiglia | Test | H₀ | N test | Note |
|---|---|---|---|---|
| **BH-A (primaria)** | Welch t + Mann-Whitney | μ_post = μ₂₀₁₉ | **16** | Split shock_hard + τ_price |
| **BH-B (ausiliaria)** | DiD IT vs DE/SE | δ_DiD = 0 | **8** | Domanda diversa → BH separata |
| Esplorativi [no BH] | Block perm + HAC | μ_post = μ_pre | 16 | H₀ locale ≠ H₀ primaria |
| Esplorativi [no BH] | τ_margin timing | — | — | Descrittivo puro |
| **TOTALE BH** | | | **24** | (vs 56 in v1) |

τ_margin è rimosso dalla BH confirmatory (era circolare). Block perm e HAC sono
esplorativi perché rispondono a "c'è un salto pre→post?" — domanda diversa da "il livello
è anomalo vs 2019?"

### Risultati BH-A (famiglia primaria, 16 test)

**16/16 rigettati a FDR 5%.**

| Evento | Carburante | Split | δ_vs_2019 | Welch p | BH adj. | n_eff | Classificazione |
|---|---|---|---|---|---|---|---|
| Ucraina | Benzina | shock_hard | +0.089 | 0.0001 | 0.0001 | **4.0 ⚠** | Inconclusivo (n_eff < 5) |
| Ucraina | Benzina | τ_price | +0.074 | 0.0000 | 0.0000 | 5.4 ⚠ | **Confermato** |
| Ucraina | Diesel | shock_hard | +0.073 | 0.0008 | 0.0009 | 8.0 ⚠ | **Confermato** |
| Ucraina | Diesel | τ_price | +0.056 | 0.0009 | 0.0009 | 10.4 | **Confermato** |
| Iran-Is. | Benzina | shock_hard | +0.079 | 0.0000 | 0.0000 | 6.3 ⚠ | **Confermato** |
| Iran-Is. | Benzina | τ_price | +0.076 | 0.0000 | 0.0000 | 7.5 ⚠ | **Confermato** |
| Iran-Is. | Diesel | shock_hard | +0.060 | 0.0000 | 0.0000 | 17.5 | **Confermato** |
| Iran-Is. | Diesel | τ_price | +0.061 | 0.0000 | 0.0000 | 14.2 | **Confermato** |

**Lettura del flag n_eff:** con ρ̂ ≈ 0.70 per Ucraina benzina, ci sono solo ~8
osservazioni indipendenti su 44 nominali (fattore inflazione 5.6×). Il test di Welch
riporta t molto alto ma l'incertezza reale è più bassa. I rigetti BH sono formalmente
corretti, ma la magnitudine dell'evidenza è sovrastimata per i casi con n_eff < 10.
Ucraina benzina shock_hard (n_eff=4.0) è classificato "Inconclusivo" nonostante BH_reject=True.

### Risultati BH-B (famiglia ausiliaria DiD, 8 test)

> ⚠ **BUG: script 05 carica `auxiliary_pvalues.csv` (v1, obsoleto) invece di
> `did_results_v2.csv` (v2).** Il report "0/8 rigettati" nel sommario è ERRATO per
> questa run. I risultati corretti da `did_results_v2.csv` mostrano 7/8 significativi
> (ma molti con δ<0 e PTA violata). Vedere §4.3 per dettagli.

### Esplorativi (block perm + HAC)

Con il τ_price come split, i test locali (salto pre→post) danno quasi sempre n.s.
per Iran-Israele (il margine era già alto prima dello shock e si è compresso durante).
Per Ucraina benzina con Andrews BW=22 (≥ n/2=22) il test HAC è quasi non informativo.
Questi risultati sono coerenti — non contraddittori — con i rigetti BH-A: un livello
elevato vs 2019 e un salto locale nullo/negativo convivono quando il pre-shock era
già anomalamente alto.

---

## PASSO 6 — n_eff, Andrews BW e diagnostica distribuzione

### Numeri effettivi di osservazioni indipendenti (script 06)

Con ρ̂ AR(1) elevato, la formula n_eff = n·(1−ρ̂)/(1+ρ̂) rivela quanto sia limitata la
vera informazione in ogni finestra post-shock:

| Evento | Carburante | n nominale | ρ̂ AR(1) | n_eff | Andrews BW | Inflazione | Valutazione |
|---|---|---|---|---|---|---|---|
| Ucraina | Benzina | 44 | 0.695 | **7.9** | 22 | 5.6× | CRITICO: BW ≥ n/2, HAC non informativo |
| Ucraina | Diesel | 44 | 0.448 | 16.8 | 9 | 2.6× | ATTENZIONE |
| Iran-Is. | Benzina | 29 | 0.652 | **6.1** | 14 | 4.7× | ATTENZIONE: BW ≥ n/2 |
| Iran-Is. | Diesel | 29 | 0.562 | 8.1 | 12 | 3.6× | ATTENZIONE |

**Implicazione pratica:** dove Andrews BW ≥ n/2, la finestra ottimale per la correzione
HAC è più larga della metà della serie stessa — il test HAC non ha abbastanza "base" per
essere informativo. Per Ucraina benzina e Iran-Is. benzina, l'evidenza utile viene dal
Mann-Whitney (robusto all'autocorrelazione) e dal confronto annuale, non dall'HAC.

### Verifica dell'assunzione distributiva

Il modello bayesiano (passo 2) usa la StudentT. Lo script 06 confronta quattro
famiglie distributive via AIC: Normale, StudentT, Skew-Normal, Skewed-T (Fernandez-Steel).

**Raccomandazioni:**

| Scenario | Raccomandazione |
|---|---|
| Ucraina Benzina/Diesel (log-prezzi, crack spread) | **Skewed-T** |
| Ucraina Brent | StudentT (ok) |
| Iran-Is. Brent, Benzina | StudentT (ok) |
| Iran-Is. Diesel (log-prezzi, crack spread) | Misto (Normale / Skewed-T) |
| Hormuz Brent, Benzina | **Skewed-T** |
| Hormuz Diesel | Normale |

La Skewed-T è raccomandata per 7/13 scenari con dati sufficienti. Questo suggerisce
che i residui sono asimmetrici: i movimenti verso l'alto sono più estremi o più
frequenti di quelli verso il basso. Per Ucraina benzina, il parametro ν=1.47 (code
quasi-Cauchy) conferma l'inadeguatezza della Normale.

**Impatto:** i test di script 03 (non parametrici o HAC) NON sono impattati.
Solo la stima precisa di τ e il suo credible interval potrebbero spostarsi
di qualche settimana con la Skewed-T. L'impatto qualitativo atteso è limitato.

---

## PASSO 7 — Anomalia strutturale pre-shock Iran-Israele (script 07)

### Il problema che script 07 affronta

Il margine italiano era già strutturalmente elevato prima dello shock di giugno 2025.
Lo script 07 indaga: **quando è iniziata questa anomalia?** usando Bai-Perron brute-force
sulla finestra 2023-01 / 2025-06-12 (prima dello shock).

### Break strutturali (Bai-Perron BIC-ottimale)

| Carburante | τ_pre | δ | Prima | Dopo | Interpretazione |
|---|---|---|---|---|---|
| Benzina | **2024-07-15** | +0.019 | 0.230 EUR/L | 0.250 EUR/L | Rialzo strutturale estate 2024 |
| Diesel | **2023-05-15** | −0.067 | 0.245 → 0.178 | (discesa) | Normalizzazione post-2022 |
| Diesel | **2023-10-02** | +0.040 | 0.178 → 0.218 | (rialzo) | Seconda rottura diesel 2023 |

Il break di benzina cade a luglio 2024, non in prossimità del conflitto Iran-Israele
(giugno 2025). Il margine era già al picco storico da ~10 mesi prima dello shock.

### Confronto annuale e H1 2025 pre-shock

| Anno / Periodo | Benzina μ (EUR/L) | δ vs 2019 | MW vs 2019 p |
|---|---|---|---|
| 2019 | 0.168 | 0 | — |
| 2023 | 0.235 | +0.067 | 0.0000 |
| 2024 | 0.233 | +0.065 | 0.0000 |
| **2025-H1 (pre-shock)** | **0.254** | **+0.086** | **0.0000** |

2025-H1 è significativamente superiore anche a 2023 e 2024 (MW p=0.004 e p=0.001),
non solo al 2019.

**Conclusione:** l'anomalia del margine per Iran-Israele è **pre-esistente e strutturale**,
iniziata tra 2023 e luglio 2024. Lo shock del giugno 2025 non l'ha creata — semmai
l'ha compresso leggermente (δ_local < 0 per entrambi i carburanti, script 03).
Questo è il risultato più robusto dell'intera analisi Iran-Israele.

> ⚠ **Tutti i test in script 07 sono ESPLORATIVI.** I p-value MW non entrano nella BH.

---

## Analisi annuale

Il margine elevato è transitorio (solo durante le crisi) o strutturale?

**Benzina:**

| Anno | Margine (EUR/L) | δ vs 2019 | Anomalo (2σ)? | Windfall M€ |
|---|---|---|---|---|
| 2019 | 0.168 | 0 | no | 0 |
| 2020 | 0.204 | +0.036 | no* | +340 |
| 2021 | 0.182 | +0.013 | no | +126 |
| 2022 | 0.254 | **+0.085** | **SÌ** | **+807** |
| 2023 | 0.235 | **+0.067** | **SÌ** | +633 |
| 2024 | 0.233 | **+0.064** | **SÌ** | +622 |
| 2025 | 0.255 | **+0.086** | **SÌ** | **+818** |
| 2026 (parz.) | 0.274 | **+0.106** | **SÌ** | +288 |

*2020: δ=+0.036 appena sotto la soglia 2σ=0.038. COVID ha ridotto i volumi di ~40%:
il windfall totale è sovrastimato dall'uso di volumi fissi 2022.

**I margini sono rimasti strutturalmente sopra il 2019 per tutti gli anni dal 2022.**
Il 2025 supera addirittura il 2022. Non si tratta di spike temporanei durante le crisi,
ma di un livello più alto stabilizzato per 4 anni consecutivi.

> ⚠ **IL 2025 È GIÀ ANOMALO PRIMA DI HORMUZ:** l'analisi annuale mostra che il 2025
> aveva il margine più alto della serie storica (benzina +0.086) prima dell'evento di
> Hormuz (feb 2026). La classificazione "Inconclusivo" per Hormuz benzina va letta in
> questo contesto: il margine era già al picco e lo shock non ha fatto molto.

**Diesel** — stesso pattern con windfall maggiori: 2022 +2.297 M€, 2025 +2.362 M€.

---

## Cosa questa analisi NON può dire

**1. La causa del margine elevato.** "Margine anomalo positivo" è coerente con: 
comportamento opportunistico, effetti FIFO/LIFO su inventario (compri quando costa
meno e vendi al prezzo di oggi), risk premium razionale durante l'incertezza,
costi operativi aumentati (assicurazioni, logistica), riduzione temporanea della
concorrenza. Separare queste cause richiederebbe dati interni non pubblici.

**2. La specificità italiana.** Il DiD dice che l'Italia non si è comportata in modo
statisticamente diverso dalla Germania e dalla Svezia. Per Ucraina, i δ_DiD IT-DE
sono addirittura negativi: la Germania aveva margini almeno pari all'Italia. Se
c'è un problema, è europeo.

**3. La certezza statistica.** Con ρ≈0.85 e n_eff≈2, i p-value del Welch t sono
sovrastimati. La robustezza reale dell'evidenza è quella del Mann-Whitney e del
permutation test — più conservativi, e per alcuni casi non rigettano.

**4. Hormuz in modo definitivo.** Solo 12 settimane di dati (aggiornamento da 7). 
Benzina: δ_vs_2019=+0.104 (z=5.5), ma n_eff=3.1 → test non informativo. Diesel:
δ_vs_2019=+0.015 (z=0.8), n_eff=1.3 → essenzialmente nessun segnale. Entrambe le serie
sono escluse dalla BH e classificate "Inconclusivo" / "Neutro". Raccomandato re-run
quando n_post ≥ 20 settimane.

---

## Glossario

| Termine | Significato |
|---|---|
| **Crack spread** | Prezzo pompa (netto tasse) − costo wholesale: proxy del margine lordo |
| **Baseline** | Anno di riferimento (2019) usato come confronto |
| **H₀ / H₁** | Ipotesi nulla (niente di speciale) / alternativa (qualcosa è cambiato) |
| **p-value** | Probabilità di questo risultato per caso, assumendo H₀ vera |
| **σ** | Deviazione standard: misura quanto variano normalmente i dati |
| **DW (Durbin-Watson)** | Misura di autocorrelazione nei residui (ok: 1.5-2.5) |
| **ρ** | Coefficiente di autocorrelazione AR(1) (qui ≈ 0.85-0.90) |
| **n_eff** | Numero effettivo di osservazioni indipendenti = n·(1-ρ)/(1+ρ) |
| **Changepoint (τ)** | Data in cui la traiettoria di una serie ha cambiato strutturalmente |
| **τ_price** | Changepoint dei log-prezzi wholesale (esogeno al margine, MCMC) |
| **τ_margin** | Changepoint del crack spread (ENDOGENO — non nella BH confirmatory) |
| **τ_lag** | τ_margin − τ_price (ANTICIPATORIO / SINCRONO / REATTIVO) |
| **MCMC** | Metodo di simulazione stocastica per stimare distribuzioni di probabilità |
| **Rhat** | Indicatore di convergenza MCMC (buono se ≤ 1.01; preoccupante > 1.05) |
| **BH (Benjamini-Hochberg)** | Correzione per test multipli che controlla il FDR |
| **FDR** | False Discovery Rate: proporzione di risultati falsi tra quelli rigettati |
| **DiD** | Difference-in-Differences: confronto tra paese trattato e gruppo di controllo |
| **PTA** | Parallel Trends Assumption: condizione di validità del DiD |
| **HAC** | Errori standard corretti per autocorrelazione e eteroschedasticità |
| **Cliff's delta** | Effect size ordinale: +1 = post sempre maggiore di pre |
| **Windfall** | Extramargine cumulato sopra baseline × volumi (proxy, non profitto netto) |
| **AIC** | Criterio di selezione del modello: più basso = meglio (penalizza complessità) |
| **Skewed-T** | Distribuzione con code pesanti E asimmetria (Fernandez-Steel) |
| **δ_local** | post − pre nella stessa finestra evento (misura il salto durante l'evento) |
| **δ_vs_2019** | post − μ₂₀₁₉ (misura il livello assoluto rispetto al baseline) |
| **pre_anomalo** | Il periodo pre-shock era già sopra la soglia 2σ del 2019 |