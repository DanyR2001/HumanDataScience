# Documentazione test statistici — `02e_statistical_tests.py`

Pipeline crack-spread IT · Prezzi carburante MIMIT/SISEN vs Futures (2015–2026)

---

## Serie analizzate

| Serie | Descrizione |
|---|---|
| `benzina_net` | Prezzo pompa benzina al netto di IVA e accise (€/L) |
| `gasolio_net` | Prezzo pompa gasolio al netto di IVA e accise (€/L) |
| `margin_benzina` | `benzina_net` − futures Eurobob €/L (margine distributore) |
| `margin_gasolio` | `gasolio_net` − futures Gas Oil €/L (margine distributore) |

## Struttura dei test

Lo script è organizzato in tre blocchi principali che operano a scale temporali diverse:

```
A. Intera serie storica (2015–2026)   → proprietà strutturali permanenti
B. Finestre pre/post evento (±40 gg)  → confronto intorno ai 3 shock
C. Cross-event (k=3 eventi insieme)   → confronto tra i 3 shock
```

Gli **eventi** considerati sono:

| ID | Shock | Data |
|---|---|---|
| Ucraina | Invasione Russia-Ucraina | 24 feb 2022 |
| Iran-Israele | Escalation Iran-Israele | 13 giu 2025 |
| Hormuz | Crisi Stretto di Hormuz | 28 feb 2026 |

---

## A — Test sull'intera serie storica

Applicati a ciascuna delle 4 serie sull'intero campione disponibile (~4 000 osservazioni giornaliere).

### A1. Statistiche descrittive

Non sono test inferenziali ma forniscono il contesto per interpretare tutti gli altri risultati.

| Statistica | Simbolo | Interpretazione |
|---|---|---|
| Media | μ | Livello medio della serie in €/L |
| Mediana | — | Robusto agli outlier; confrontato con μ indica asimmetria |
| Deviazione standard | σ | Dispersione assoluta |
| Coefficiente di variazione | CV = σ/μ | Dispersione relativa, confrontabile tra serie a scale diverse |
| Skewness | γ₁ | > 0 → coda destra (shock al rialzo più frequenti) |
| Curtosi (excess) | γ₂ | > 0 → code più pesanti della normale (leptocurtica) |
| IQR | Q₃ − Q₁ | Range interquartile, misura di spread robusta |
| Percentili 5/95 | p₅, p₉₅ | Estremi della distribuzione centrale |
| φ AR(1) | φ | Autocorrelazione lag-1; usata per calcolare n_eff |
| n_eff | n · (1−φ)/(1+φ) | Dimensione effettiva del campione corretta per autocorrelazione |

> **Nota su n_eff.** Con serie giornaliere di prezzi, φ è tipicamente 0.95–0.99. Questo riduce la dimensione effettiva da ~4 000 a poche decine di osservazioni indipendenti, rendendo i test parametrici standard conservativi o invalidi. Per questo motivo nei test sulle finestre viene applicata la correzione n_eff sistematicamente.

---

### A2. Stazionarietà

La stazionarietà (media e varianza costanti nel tempo) è un prerequisito fondamentale per la maggior parte dei modelli ITS. I due test vengono usati in combinazione perché hanno ipotesi nulle opposte.

#### ADF — Augmented Dickey-Fuller

- **H₀:** la serie ha una radice unitaria (è *non* stazionaria, random walk)
- **H₁:** la serie è stazionaria
- **Selezione lag:** automatica via AIC per gestire l'autocorrelazione residua
- **Decisione:** p < α → rifiuto H₀ → serie stazionaria
- **Limite:** ha bassa potenza contro alternative vicine alla radice unitaria (es. AR con φ = 0.98)

#### KPSS — Kwiatkowski-Phillips-Schmidt-Shin

- **H₀:** la serie è stazionaria (intorno a una costante o un trend)
- **H₁:** la serie ha una radice unitaria
- **Varianti:** `level` (stazionarietà intorno a costante) e `trend` (intorno a trend lineare)
- **Decisione:** p < α → rifiuto H₀ → serie *non* stazionaria
- **Limite:** tende a rifiutare H₀ spesso su serie lunghe (dimensione distorta)

#### Verdetto duale ADF + KPSS

| ADF | KPSS | Conclusione |
|---|---|---|
| Rifiuta H₀ (p < α) | Non rifiuta H₀ (p ≥ α) | **STAZIONARIA** — entrambi concordano |
| Non rifiuta H₀ | Rifiuta H₀ | **NON STAZIONARIA** — entrambi concordano |
| Rifiuta H₀ | Rifiuta H₀ | **INCERTO** — conflitto; possibile stazionarietà di lungo periodo o struttura non lineare |
| Non rifiuta H₀ | Non rifiuta H₀ | **INCERTO** — campione insufficiente |

---

### A3. Normalità

Tutti i test verificano se la distribuzione della serie è compatibile con una Normale. Applicati sulla serie grezza (non sulle differenze), catturano la forma della distribuzione incondizionata.

#### Jarque-Bera

- **H₀:** skewness = 0 e curtosi excess = 0 (distribuzione normale)
- **Statistica:** JB = n/6 · [γ₁² + γ₂²/4], distribuita asintoticamente χ²(2)
- **Punto di forza:** asintoticamente valido per n grandi, è il test primario su serie lunghe
- **Limite:** con n molto grande (> 1 000) tende a rifiutare H₀ anche per deviazioni trascurabili dalla normalità

#### D'Agostino K²

- **H₀:** la distribuzione è normale
- **Statistica:** combina z-score di skewness e curtosi con una trasformazione che migliora l'approssimazione normale per n moderati
- **Punto di forza:** più potente di JB per n ∈ [20, 300]; usato come test secondario
- **Limite:** meno efficiente di JB per n molto grandi

#### Shapiro-Wilk

- **H₀:** il campione proviene da una distribuzione normale
- **Punto di forza:** il test più potente per n piccoli (< 50); ha la migliore potenza uniforme
- **Limite:** computazionalmente inadatto per n > 200; lo script lo applica **solo se n ≤ 200**

---

### A4. Autocorrelazione

L'autocorrelazione indica che i valori passati contengono informazione sui valori futuri. È quasi sempre presente nelle serie finanziarie giornaliere e viola l'ipotesi i.i.d. dei test parametrici standard.

#### Ljung-Box

- **H₀:** i primi *k* autocorrelazioni sono congiuntamente nulle (rumore bianco)
- **Statistica:** Q = n(n+2) Σ [ρ̂ₖ²/(n−k)], distribuita approssimativamente χ²(k)
- **Lag testati:** 5, 10, 20 (per catturare autocorrelazione a breve, media e lunga distanza)
- **Decisione:** p < α → rifiuto H₀ → autocorrelazione significativa
- **Applicazione:** un LB significativo a lag 5 e 10 implica che i modelli OLS standard producono errori standard distorti → necessità di correzione HAC o AR

#### Durbin-Watson

- **Statistica:** DW = Σ(eₜ − eₜ₋₁)² / Σeₜ², con range [0, 4]
- **Interpretazione:** DW ≈ 2 → no autocorrelazione; DW < 2 → autocorrelazione positiva; DW > 2 → autocorrelazione negativa
- **Limite:** testa solo l'autocorrelazione di primo ordine (lag 1), non è un test formale (non ha distribuzione nulla esatta per serie generali)

---

### A5. Effetti ARCH

Gli effetti ARCH (Autoregressive Conditional Heteroscedasticity) indicano che la **varianza** della serie non è costante nel tempo ma si raggruppa: periodi di alta volatilità tendono a seguire periodi di alta volatilità (clustering). Sono tipici dei prezzi energetici.

#### Engle ARCH-LM

- **H₀:** nessun effetto ARCH fino al lag *k* (varianza condizionale costante)
- **Procedura:** regredisce i residui al quadrato (eₜ²) sui loro valori ritardati; la statistica LM = n · R² è distribuita χ²(k)
- **Lag testati:** 5 e 10
- **Decisione:** p < α → rifiuto H₀ → presenza di effetti ARCH
- **Implicazione pratica:** se ARCH è presente, il test di Levene per l'uguaglianza delle varianze è inaffidabile (segnalato nella sezione B3)

---

### A6. Memoria lunga — Esponente di Hurst

L'esponente di Hurst H misura se la serie ha **memoria lunga**: i valori passati influenzano quelli futuri oltre le scale di breve periodo catturate dall'autocorrelazione lag-1.

**Metodo usato: R/S (Rescaled Range)**

La serie viene suddivisa in segmenti di lunghezza *n*. Per ogni segmento si calcola R/S = (range cumulato) / (deviazione standard). La relazione E[R/S] ~ nᴴ fornisce H tramite regressione log-log.

| Valore di H | Interpretazione |
|---|---|
| H ≈ 0.5 | Random walk — nessuna memoria; incrementi non correlati |
| H > 0.5 | Persistenza — un rialzo tende a essere seguito da un rialzo (memoria lunga) |
| H < 0.5 | Anti-persistenza — mean-reverting; la serie tende a invertire la direzione |

Lo script calcola anche un **Hurst rolling** con finestra mobile di 250 giorni per visualizzare come il regime di memoria cambia nel tempo, in particolare intorno agli shock geopolitici.

---

## B — Test sulle finestre evento

Per ogni combinazione *serie × evento* (4 serie × 3 eventi = 12 celle) si estrae:

- **Pre-shock:** i 40 giorni immediatamente precedenti la data di shock
- **Post-shock:** i 40 giorni immediatamente successivi

I test confrontano le due finestre per rilevare cambiamenti strutturali nel livello, nella dispersione e nella forma della distribuzione.

**Ipotesi nulla di riferimento (H₀):** In prossimità degli shock geopolitici, i distributori italiani non generano profitti anomali — gli aumenti dei prezzi sono coerenti con i costi di approvvigionamento.

---

### B1. Statistiche descrittive per finestra

Identiche a quelle della sezione A, calcolate separatamente su pre e post. Vengono aggiunte:

| Statistica | Formula | Significato |
|---|---|---|
| Δmedia | μ_post − μ_pre | Variazione assoluta del livello (€/L) |
| Δ% | (μ_post − μ_pre) / \|μ_pre\| × 100 | Variazione percentuale |
| σ-ratio | σ_post / σ_pre | Rapporto delle volatilità; > 1 indica aumento di volatilità post-shock |

---

### B2. Uguaglianza delle medie

Questi tre test rispondono alla domanda: **il livello medio è significativamente diverso tra pre e post?**

Usati in combinazione per un **verdetto a maggioranza**: se ≥ 2 test su 3 rifiutano H₀ → `ANOMALO`, altrimenti `NON_ANOMALO`.

#### Welch t-test con correzione n_eff

- **H₀:** μ_pre = μ_post
- **H₁ (one-sided):** μ_post > μ_pre
- **Innovazione:** i gradi di libertà non usano n reale ma **n_eff = n · (1−φ)/(1+φ)**, dove φ è l'autocorrelazione AR(1). Con φ = 0.9, n_eff ≈ n/19. Questo rende il test conservativo ma valido in presenza di autocorrelazione
- **Limite:** assume che la differenza delle medie sia la statistica rilevante; sensibile a distribuzioni molto asimmetriche

#### Mann-Whitney U (one-sided)

- **H₀:** la distribuzione di post è identica a quella di pre
- **H₁:** i valori di post tendono ad essere più grandi di quelli di pre (stochastic dominance)
- **Punto di forza:** non parametrico — non assume normalità né omoschedasticità; robusto a outlier e distribuzioni asimmetriche
- **Statistica:** U = numero di coppie (post_i, pre_j) in cui post_i > pre_j
- **Limit:** non testa specificamente la media ma l'ordinamento stocastico

#### Wilcoxon signed-rank (one-sided)

- **H₀:** la mediana delle differenze (post − pre) è zero
- **H₁:** la mediana è positiva (post > pre)
- **Procedura:** le differenze paired vengono ordinate per valore assoluto e si somma il rango dei positivi
- **Nota implementativa:** poiché le finestre pre e post hanno spesso lunghezze diverse, si prendono i min(n_pre, n_post) elementi più recenti di pre e i più recenti di post per formare le coppie
- **Punto di forza:** più potente di Mann-Whitney quando le differenze hanno una direzione consistente

---

### B3. Uguaglianza delle varianze

Questi test rispondono alla domanda: **la volatilità è cambiata tra pre e post?**

Usati in combinazione per il verdetto: se ≥ 2 test su 3 rifiutano H₀ → `VOLATILITA_AUMENTATA` o `VOLATILITA_RIDOTTA` (in base al segno di σ-ratio), altrimenti `VOLATILITA_STABILE`.

> **Attenzione:** in presenza di effetti ARCH (rilevati nella sezione A5), la varianza condizionale non è costante neanche all'interno delle singole finestre. I test di seguito assumono varianza costante all'interno di ciascun gruppo; i loro risultati devono essere interpretati con cautela.

#### Levene

- **H₀:** Var(pre) = Var(post)
- **Procedura:** ANOVA sulla statistica |xᵢ − μ̄|, dove μ̄ è la media del gruppo
- **Punto di forza:** robusto a deviazioni dalla normalità (usa la media delle deviazioni assolute, non le varianze)

#### Brown-Forsythe

- **H₀:** Var(pre) = Var(post)
- **Come Levene ma usa la mediana** invece della media: |xᵢ − mediana|
- **Punto di forza:** più robusto di Levene in presenza di distribuzioni asimmetriche e outlier
- **Nota:** è implementato come `levene(..., center="median")` in scipy

#### Fligner-Killeen

- **H₀:** Var(pre) = Var(post)
- **Procedura:** test basato sui ranghi delle deviazioni dalla mediana; completamente non parametrico
- **Punto di forza:** il più robusto tra i tre in presenza di non-normalità; preferito come test primario quando JB rifiuta la normalità

#### Bartlett

- **H₀:** Var(pre) = Var(post)
- **Punto di forza:** massima potenza quando i dati sono normali
- **Limite critico:** molto sensibile alla non-normalità — la statistica di test si gonfia artificialmente se i dati non sono normali. Usato solo come riferimento comparativo

#### F-ratio test

- **H₀:** Var(pre) = Var(post)
- **Statistica:** F = Var(pre)/Var(post), distribuita F(n₁−1, n₂−1) sotto H₀
- **Test bilaterale:** p = 2 · min(CDF(F), 1−CDF(F))
- **Limite:** estremamente sensibile alla non-normalità; riportato per completezza ma da interpretare con cautela

---

### B4. Uguaglianza della distribuzione

Questi test verificano se l'**intera distribuzione** è cambiata, non solo media o varianza.

#### KS 2-campioni (Kolmogorov-Smirnov)

- **H₀:** le due distribuzioni sono identiche (F_pre = F_post)
- **Statistica:** D = max|F̂_pre(x) − F̂_post(x)| — la massima distanza tra le CDF empiriche
- **Punto di forza:** rileva qualsiasi tipo di cambiamento (media, varianza, forma, code)
- **Limite:** bassa potenza su campioni piccoli (n < 30); sensibile principalmente al centro della distribuzione

#### Anderson-Darling 2-campioni

- **H₀:** le due campioni provengono dalla stessa distribuzione
- **Punto di forza:** maggiore peso alle code rispetto al KS — più sensibile a differenze nelle distribuzioni delle code, dove si concentrano gli shock estremi dei prezzi energetici
- **Nota sul p-value:** quando la statistica è molto estrema, scipy non riesce a calcolare il p-value esattamente tramite il metodo analitico standard e lo "floora" a 0.001. Il valore riportato nel CSV è corretto (p < 0.001); il warning viene soppresso silenziosamente perché non indica un errore

---

### B5. Effect size

I test di significatività dipendono da n. Con n sufficientemente grande qualsiasi differenza risulta significativa. Le misure di effect size quantificano la **rilevanza pratica** della differenza indipendentemente dalla dimensione del campione.

#### Cohen d

$$d = \frac{\mu_{post} - \mu_{pre}}{s_p}$$

dove $s_p = \sqrt{\frac{(n_{pre}-1)s_{pre}^2 + (n_{post}-1)s_{post}^2}{n_{pre}+n_{post}-2}}$ è la deviazione standard pooled.

| |d| | Interpretazione |
|---|---|
| < 0.2 | negligible |
| 0.2 – 0.5 | small |
| 0.5 – 0.8 | medium |
| ≥ 0.8 | large |

#### Hedge g

Identico a Cohen d ma con un fattore di correzione per la distorsione da campioni piccoli:

$$g = d \cdot \left(1 - \frac{3}{4(n_{pre}+n_{post}-2)-1}\right)$$

Preferibile a Cohen d quando n < 20; converge a d per n grandi.

#### Cliff δ (delta)

$$\delta = \frac{\#(post_i > pre_j) - \#(post_i < pre_j)}{n_{pre} \cdot n_{post}}$$

- Range: [−1, +1]
- Completamente non parametrico — non assume normalità né scala di misura
- Interpretazione diretta: δ = +0.8 significa che l'80% delle coppie (post, pre) vede post > pre
- Usa le stesse soglie di Cohen (0.147, 0.330, 0.474) adattate alla scala [−1, +1]

> **Cliff δ è la misura di effect size primaria** in questo script perché non richiede normalità ed è direttamente interpretabile come probabilità di superiorità stocastica, coerente con il Mann-Whitney U.

---

### B6. Normalità intra-finestra

Shapiro-Wilk e Jarque-Bera applicati separatamente su pre e post (stesse specifiche della sezione A3, con n piccoli sempre < 200 per cui SW è sempre calcolato). Servono per contestualizzare la validità dei test parametrici (Welch, Bartlett) applicati alle finestre.

---

### B7. Stazionarietà intra-finestra

ADF applicato separatamente su pre e post. Con n ≈ 40 la potenza è bassa (note come "finestre corte"), ma un ADF non significativo all'interno della finestra pre indica che il periodo di riferimento ha un trend, il che invaliderebbe l'assunzione baseline-piatta dei modelli ITS naïve.

---

### B8. Autocorrelazione intra-finestra

Ljung-Box a lag 5 applicato su pre e post separatamente. Un LB significativo all'interno della finestra indica che le osservazioni non sono indipendenti, il che:

1. Riduce ulteriormente n_eff rispetto al calcolo globale
2. Giustifica l'uso di modelli con correzione HAC (Newey-West) o ARIMAX invece di OLS semplice

---

## C — Confronto cross-evento

Per ogni serie e per tre fasi temporali (`pre`, `post`, `delta = post − media_pre`), i 3 eventi vengono confrontati simultaneamente come **k = 3 campioni indipendenti**.

L'obiettivo è verificare se i tre shock producono effetti strutturalmente diversi o se il meccanismo di trasmissione è stabile tra eventi.

### C1. Kruskal-Wallis

- **H₀:** le k distribuzioni sono identiche (stesso centro)
- **Generalizzazione non parametrica dell'ANOVA** — non assume normalità
- **Statistica:** H = 12/(n(n+1)) · Σ nⱼ(R̄ⱼ − (n+1)/2)², dove R̄ⱼ è il rango medio del gruppo j
- **Punto di forza:** robusto a non-normalità e eteroschedasticità; test primario
- **Limite:** testa le medie dei ranghi, non le medie originali

### C2. ANOVA a una via

- **H₀:** μ₁ = μ₂ = μ₃ (le medie dei 3 eventi sono uguali)
- **Statistica:** F = varianza tra gruppi / varianza entro gruppi
- **Punto di forza:** massima potenza se i dati sono normali e omoschedastici
- **Limite:** assume normalità e omoschedasticità; riportato per confronto con Kruskal-Wallis
- **Nota:** un disaccordo KW ns / ANOVA * suggerisce che la normalità o l'omoschedasticità sono violate

### C3. Fligner-Killeen k-campioni

- **H₀:** Var(gruppo₁) = Var(gruppo₂) = Var(gruppo₃)
- Estensione non parametrica del test delle varianze a k > 2 gruppi
- Stesse proprietà di robustezza della versione a 2 campioni (sezione B3)

### C4. Bartlett k-campioni

- **H₀:** Var(gruppo₁) = Var(gruppo₂) = Var(gruppo₃)
- Estensione del test Bartlett a k > 2 gruppi; assume normalità
- Riportato per confronto con Fligner-Killeen

---

## Output prodotti

| File | Contenuto |
|---|---|
| `stat_tests_global.csv` | Una riga per serie, tutte le statistiche e p-value della sezione A |
| `stat_tests_windows.csv` | Una riga per (serie × evento), tutti i test della sezione B |
| `stat_tests_crossevent.csv` | Una riga per (serie × fase), tutti i test della sezione C |
| `plot_windows_{serie}.png` | Per ogni serie: serie temporale + box/strip plot per i 3 eventi |
| `plot_hurst_rolling.png` | Hurst esponente rolling (finestra 250 gg) per tutte le serie |

---

## Guida alla lettura dei verdetti

### Verdetto medie (`verdict_mean_shift`)

Basato su voto di maggioranza tra Welch n_eff, Mann-Whitney e Wilcoxon:

| Voti ≥ 2/3 che rifiutano H₀ | Verdetto |
|---|---|
| Sì | `ANOMALO` — il livello post è significativamente più alto |
| No (con ≥ 2 test validi) | `NON_ANOMALO` |
| Meno di 2 test validi | `INDETERMINATO` |

### Verdetto varianze (`verdict_var_shift`)

Basato su voto di maggioranza tra Levene, Fligner-Killeen e Brown-Forsythe:

| Voti ≥ 2/3 rifiutano H₀ | σ_post vs σ_pre | Verdetto |
|---|---|---|
| Sì | post > pre | `VOLATILITA_AUMENTATA` |
| Sì | post < pre | `VOLATILITA_RIDOTTA` |
| No | — | `VOLATILITA_STABILE` |

### Convenzione p-value

| Simbolo | Significato |
|---|---|
| `***` | p < 0.001 |
| `**` | p < 0.01 |
| `*` | p < 0.05 |
| `ns` | p ≥ 0.05 (non significativo) |

---

## Riferimenti

- **Hurst (1951):** Long-term storage capacity of reservoirs. *Trans. Am. Soc. Civ. Eng.* 116, 770–808
- **Dickey & Fuller (1979):** Distribution of the estimators for autoregressive time series with a unit root. *JASA* 74, 427–431
- **Kwiatkowski et al. (1992):** Testing the null hypothesis of stationarity against the alternative of a unit root. *J. Econometrics* 54, 159–178
- **Engle (1982):** Autoregressive conditional heteroscedasticity. *Econometrica* 50, 987–1007
- **Ljung & Box (1978):** On a measure of lack of fit in time series models. *Biometrika* 65, 297–303
- **Mann & Whitney (1947):** On a test of whether one of two random variables is stochastically larger. *Ann. Math. Stat.* 18, 50–60
- **Cliff (1993):** Dominance statistics: Ordinal analyses. *Psychol. Bull.* 114, 494–509
- **Cohen (1988):** *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.)
- **Levene (1960):** Robust tests for equality of variances. *Contributions to Probability and Statistics*, 278–292
- **Fligner & Killeen (1976):** Distribution-free two-sample tests for scale. *JASA* 71, 210–213
- **Anderson & Darling (1952):** Asymptotic theory of certain goodness-of-fit criteria. *Ann. Math. Stat.* 23, 193–212